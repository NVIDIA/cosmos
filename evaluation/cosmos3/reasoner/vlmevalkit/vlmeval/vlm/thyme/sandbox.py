import ast
import contextlib
import glob
import importlib
import io
import json
import math
import os
import pickle
import random
import re
import shutil
import stat
import string
import sys
import textwrap
import types

import numpy
import timeout_decorator  # For efficient timeout control
from PIL import Image, UnidentifiedImageError

# Attempt to import cv2, but make it optional if not strictly needed by
# all sandbox uses
try:
    import cv2
except ImportError:
    cv2 = None  # Keep track if cv2 is not available
    print(
        "Warning: OpenCV (cv2) not found. cv2-dependent sandboxed code will fail.",
        file=sys.stderr,
    )


# --- Enhanced Security: Prohibit dangerous system calls ---
DANGEROUS_PATTERNS = [
    r"\bsys\.",
    r"\bsocket\.",
    r"\bsubprocess\.",
    r"\bexec\(",
    r"\beval\(",
    r"\bcompile\(",
    r"\b__import__\(",
    r"\bos\.(remove|unlink|rmdir)\b",
    r"\bshutil\.rmtree\b",
    r"\bshutil\.move\b",
    r"\bos\.(rename|renames)\b",
]


def check_dangerous_code(code_string):
    """
    Performs a basic static analysis to detect dangerous patterns in the code.
    Returns True if dangerous patterns are found, False otherwise.
    """
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, code_string, re.IGNORECASE):
            return True
    return False


# pandayin: set a time limit to avoid dead loop, e.x., while(1) type loops.
EXEC_TIME_LIMIT = 10
# Define the temporary folder for processed images
# IMPORTANT: Ensure this directory exists and is writable by the script.
# For local testing, you might want to change this, e.g.:
TEMP_PROCESSED_IMAGES_DIR = "./temp_processed_images/"


class ReadOnlyPath:
    def __init__(self, path):
        self.path = path if isinstance(path, str) else None
        self.original_permissions = None

    def __enter__(self):
        if self.path and os.path.isfile(self.path):
            try:
                self.original_permissions = os.stat(self.path).st_mode
                read_only_permissions = self.original_permissions & ~(
                    stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
                )
                if self.original_permissions != read_only_permissions:
                    os.chmod(self.path, read_only_permissions)
            except OSError as e:
                print(
                    f"Warning: Could not make '{self.path}' read-only: {e}",
                    file=sys.stderr,
                )
                self.original_permissions = None  # 确保我们不会在退出时尝试恢复权限
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if (
            self.path
            and self.original_permissions is not None
            and os.path.isfile(self.path)
        ):
            try:
                os.chmod(self.path, self.original_permissions)
            except OSError as e:
                print(
                    f"Warning: Could not restore original permissions for '{self.path}': {e}",
                    file=sys.stderr,
                )


def align_first_line_to_second(code_string: str) -> str:
    lines = code_string.splitlines()

    first_line_info = None
    second_line_info = None

    for index, line_content in enumerate(lines):
        if line_content.strip():
            if first_line_info is None:
                first_line_info = {"index": index, "content": line_content}
            elif second_line_info is None:
                second_line_info = {"index": index, "content": line_content}
                break

    if not first_line_info or not second_line_info:
        return code_string

    first_line_content = first_line_info["content"]
    second_line_content = second_line_info["content"]

    first_line_indent = " " * (
        len(first_line_content) - len(first_line_content.lstrip(" "))
    )
    second_line_indent = " " * (
        len(second_line_content) - len(second_line_content.lstrip(" "))
    )

    if first_line_indent != second_line_indent:
        original_index = first_line_info["index"]
        stripped_content = first_line_content.lstrip(" ")
        lines[original_index] = second_line_indent + stripped_content

    return "\n".join(lines)


def get_image_paths(temp_output_dir):
    extensions = ["jpg", "jpeg", "png", "bmp", "gif", "tiff"]
    image_paths = []

    for ext in extensions:
        pattern = os.path.join(temp_output_dir, f"*.{ext}")
        image_paths.extend(glob.glob(pattern))

    return image_paths


VALID_IMAGE_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".tiff",
    ".webp")
ORIGINAL_OUTPUT_PREFIX = "/mnt/data/temp_processed_images/"


class ImagePathTransformer(ast.NodeTransformer):
    def __init__(self, replacement_path):
        self.replacement_path = replacement_path
        self.path_was_replaced = False

    def visit_Assign(self, node):
        # Process simple assignments: target_variable = "string_literal"
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target_variable_name = node.targets[0].id

            if target_variable_name == "image_path":  # Target specific variable name
                current_path_value = None
                # Extract string value from ast.Constant (Python 3.8+) or
                # ast.Str (older Python)
                if isinstance(node.value, ast.Constant) and isinstance(
                    node.value.value, str
                ):
                    current_path_value = node.value.value
                elif isinstance(node.value, ast.Str):  # For Python < 3.8
                    current_path_value = node.value.s

                if current_path_value is not None:
                    is_valid_image_on_disk = False
                    if (
                        os.path.exists(current_path_value)
                        and os.path.isfile(current_path_value)
                        and any(
                            current_path_value.lower().endswith(ext)
                            for ext in VALID_IMAGE_EXTENSIONS
                        )
                    ):
                        is_valid_image_on_disk = True

                    if not is_valid_image_on_disk:
                        # Replace the value of the AST node
                        if hasattr(ast, "Constant"):  # Python 3.8+
                            node.value = ast.Constant(
                                value=self.replacement_path)
                        else:  # Python < 3.8
                            node.value = ast.Str(s=self.replacement_path)
                        self.path_was_replaced = True

        return self.generic_visit(node)  # Continue traversing


MIN_CROP_DIMENSION = 64

# Define a minimum dimension for the crop
# MIN_CROP_DIMENSION = 64 # Duplicate definition


class CropCoordinateTransformer(ast.NodeTransformer):
    def __init__(self, image_width, image_height,
                 main_image_variable_name="image"):
        self.image_width = image_width
        self.image_height = image_height
        self.main_image_variable_name = main_image_variable_name  # New parameter
        self.coordinates_clamped = False
        # Add specific variable names to look for
        self.known_coord_var_names = [
            "crop_box",
            "bbox",
            "box",
            "coordinates",
            "coords",
            "crop_coords",
            "left_crop_coords",
            "right_crop_coords",
            "top_crop_coords",
            "bottom_crop_coords",
            # Add any other common names you anticipate
        ]

    def _get_numeric_value(self, node_value):
        if isinstance(node_value, ast.Constant) and isinstance(
            node_value.value, (int, float)
        ):
            return node_value.value
        elif isinstance(node_value, ast.Num):  # Python < 3.8
            return node_value.n
        return None

    def _create_numeric_node(self, value):
        int_value = int(round(value))
        if hasattr(ast, "Constant"):  # Python 3.8+
            return ast.Constant(value=int_value)
        else:  # Python < 3.8
            return ast.Num(n=int_value)

    def _clamp_coordinates(self, v_x1, v_y1, v_x2, v_y2):
        img_w, img_h = float(self.image_width), float(self.image_height)

        cx1 = min(v_x1, v_x2)
        cx2 = max(v_x1, v_x2)
        cy1 = min(v_y1, v_y2)
        cy2 = max(v_y1, v_y2)

        final_x1 = max(0.0, cx1)
        final_x1 = min(final_x1, img_w - 1.0 if img_w > 0 else 0.0)
        final_x2 = max(0.0, cx2)
        final_x2 = min(final_x2, img_w)
        if final_x1 >= final_x2 and img_w > 0:
            final_x1 = max(0.0, final_x2 - 1.0)
            if final_x2 <= final_x1:
                final_x2 = min(img_w, final_x1 + 1.0)

        final_y1 = max(0.0, cy1)
        final_y1 = min(final_y1, img_h - 1.0 if img_h > 0 else 0.0)
        final_y2 = max(0.0, cy2)
        final_y2 = min(final_y2, img_h)
        if final_y1 >= final_y2 and img_h > 0:
            final_y1 = max(0.0, final_y2 - 1.0)
            if final_y2 <= final_y1:
                final_y2 = min(img_h, final_y1 + 1.0)

        current_width = final_x2 - final_x1
        if img_w > 0 and current_width < MIN_CROP_DIMENSION:
            if img_w < MIN_CROP_DIMENSION:
                final_x1 = 0.0
                final_x2 = img_w
            else:
                final_x2 = final_x1 + MIN_CROP_DIMENSION
                if final_x2 > img_w:
                    final_x2 = img_w
                    final_x1 = max(0.0, final_x2 - MIN_CROP_DIMENSION)

        current_height = final_y2 - final_y1
        if img_h > 0 and current_height < MIN_CROP_DIMENSION:
            if img_h < MIN_CROP_DIMENSION:
                final_y1 = 0.0
                final_y2 = img_h
            else:
                final_y2 = final_y1 + MIN_CROP_DIMENSION
                if final_y2 > img_h:
                    final_y2 = img_h
                    final_y1 = max(0.0, final_y2 - MIN_CROP_DIMENSION)

        if img_w > 0 and final_x1 >= final_x2:
            final_x2 = min(img_w, final_x1 + 1.0)
        if img_h > 0 and final_y1 >= final_y2:
            final_y2 = min(img_h, final_y1 + 1.0)

        return [final_x1, final_y1, final_x2, final_y2]

    def visit_Assign(self, node):
        processed_node_values = False

        # Case 1: Direct unpacking e.g., x1, y1, x2, y2 = (v1, v2, v3, v4)
        # or var1, var2, var3, var4 = (v1, v2, v3, v4) if we assume any 4-tuple assign is coords.
        # For safety, let's stick to explicit x1,y1,x2,y2 or similar, or rely
        # on Case 2 for named tuples.
        if (
            len(node.targets) == 1
            and isinstance(node.targets[0], ast.Tuple)
            and len(node.targets[0].elts) == 4
            and all(isinstance(t, ast.Name) for t in node.targets[0].elts)
            and isinstance(node.value, ast.Tuple)
            # Optional: check if target names match common patterns like x1,y1,x2,y2
            # For now, any 4-tuple assignment to 4 variables is considered.
            and len(node.value.elts) == 4
        ):

            target_names = [t.id.lower() for t in node.targets[0].elts]
            # A heuristic: if names like 'x1', 'y1', 'x2', 'y2' or 'left',
            # 'top', 'right', 'bottom' are used
            is_likely_coords_unpacking = any(
                name in target_names
                for name in ["x1", "y1", "x2", "y2", "left", "top", "right", "bottom"]
            )

            if (
                is_likely_coords_unpacking
            ):  # Process if it seems like coordinate unpacking
                raw_coords_from_code = [
                    self._get_numeric_value(v) for v in node.value.elts
                ]

                if all(c is not None for c in raw_coords_from_code):
                    # Assume order from the tuple assignment e.g. x1,y1,x2,y2
                    current_x1, current_y1, current_x2, current_y2 = (
                        raw_coords_from_code
                    )
                    clamped_values = self._clamp_coordinates(
                        current_x1, current_y1, current_x2, current_y2
                    )

                    original_rounded = [int(round(c))
                                        for c in raw_coords_from_code]
                    final_rounded = [int(round(c)) for c in clamped_values]

                    if final_rounded != original_rounded:
                        self.coordinates_clamped = True
                        for i in range(4):
                            node.value.elts[i] = self._create_numeric_node(
                                clamped_values[i]
                            )
                    processed_node_values = True

        # Case 2: Assignment to a known variable name, e.g., crop_box = (v1, v2, v3, v4)
        # This will now specifically catch 'left_crop_coords = ...' and 'right_crop_coords = ...'
        # if they are in self.known_coord_var_names.
        elif (
            not processed_node_values
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            # Check against the list
            and node.targets[0].id in self.known_coord_var_names
            and isinstance(node.value, ast.Tuple)
            and len(node.value.elts) == 4
        ):

            raw_coords_from_code = [
                self._get_numeric_value(v) for v in node.value.elts]

            if all(c is not None for c in raw_coords_from_code):
                current_x1, current_y1, current_x2, current_y2 = raw_coords_from_code
                clamped_values = self._clamp_coordinates(
                    current_x1, current_y1, current_x2, current_y2
                )

                original_rounded = [int(round(c))
                                    for c in raw_coords_from_code]
                final_rounded = [int(round(c)) for c in clamped_values]

                if final_rounded != original_rounded:
                    self.coordinates_clamped = True
                    for i in range(4):
                        node.value.elts[i] = self._create_numeric_node(
                            clamped_values[i]
                        )

        # Case 3: Direct slicing assignment, e.g., cropped_var =
        # image_var[y1:y2, x1:x2]
        if (
            len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)  # e.g. sony_crop = ...
            # ... image[...]
            and isinstance(node.value, ast.Subscript)
            and isinstance(node.value.value, ast.Name)  # ... image variable ...
            and node.value.value.id == self.main_image_variable_name  # ... is the one we care about
            and isinstance(node.value.slice, ast.Index)  # ... [<slice_tuple>] ...
            and isinstance(node.value.slice.value, ast.Tuple)  # ... (slice_y, slice_x)
            and len(node.value.slice.value.elts) == 2
            and all(isinstance(s, ast.Slice) for s in node.value.slice.value.elts)
        ):  # both are Slices

            slice_tuple_node = node.value.slice.value
            y_slice_node = slice_tuple_node.elts[0]  # First slice is y-dim
            x_slice_node = slice_tuple_node.elts[1]  # Second slice is x-dim

            # Extract numeric values, converting None to 0 or full dimension
            y1_val = self._get_numeric_value(y_slice_node.lower)
            if y1_val is None and y_slice_node.lower is None:
                y1_val = 0.0  # If original was None, treat as 0

            y2_val = self._get_numeric_value(y_slice_node.upper)
            if y2_val is None and y_slice_node.upper is None:
                # If original was None, treat as full height
                y2_val = float(self.image_height)

            x1_val = self._get_numeric_value(x_slice_node.lower)
            if x1_val is None and x_slice_node.lower is None:
                x1_val = 0.0

            x2_val = self._get_numeric_value(x_slice_node.upper)
            if x2_val is None and x_slice_node.upper is None:
                x2_val = float(self.image_width)

            # If any original value was not a number or None (e.g. a variable
            # name in slice), skip
            if not all(isinstance(v, (int, float))
                       for v in [y1_val, y2_val, x1_val, x2_val]):
                # print(f"INFO: Skipping slice clamping for '{node.targets[0].id}' due to non-numeric slice parts.")
                return self.generic_visit(node)

            # Keep this order for comparison
            # original_coords = [x1_val, y1_val, x2_val, y2_val]
            clamped_x1, clamped_y1, clamped_x2, clamped_y2 = self._clamp_coordinates(
                x1_val, y1_val, x2_val, y2_val)

            final_coords_for_ast = [
                clamped_y1,
                clamped_y2,
                clamped_x1,
                clamped_x2,
            ]  # y first for slice node update
            original_coords_for_ast_compare = [y1_val, y2_val, x1_val, x2_val]

            # Round for comparison
            original_rounded = [int(round(c))
                                for c in original_coords_for_ast_compare]
            final_rounded = [int(round(c)) for c in final_coords_for_ast]

            if final_rounded != original_rounded:
                self.coordinates_clamped = True
                # Update the AST Slice nodes
                y_slice_node.lower = self._create_numeric_node(clamped_y1)
                y_slice_node.upper = self._create_numeric_node(clamped_y2)
                x_slice_node.lower = self._create_numeric_node(clamped_x1)
                x_slice_node.upper = self._create_numeric_node(clamped_x2)

        return self.generic_visit(node)  # Important to call for other nodes


class OpenCVNamespaceTransformer(ast.NodeTransformer):
    """
    Transforms incorrect OpenCV namespace calls (e.g., cv., cv4.) to cv2.
    """

    def __init__(self, incorrect_prefixes=None, correct_prefix="cv2"):
        if incorrect_prefixes is None:
            self.incorrect_prefixes = [
                "cv",
                "cv4",
                "cv3",
                "CV",
            ]  # Common incorrect ones
        else:
            self.incorrect_prefixes = incorrect_prefixes
        self.correct_prefix = correct_prefix
        self.namespace_updated = False

    def visit_Attribute(self, node):
        """
        Called for attribute access like object.attribute or object.sub_object.attribute
        We are interested in cases like:
        - cv.imread(...) -> node.value is Name(id='cv'), node.attr is 'imread'
        - cv.some_module.CONSTANT -> node.value is Name(id='cv'), node.attr is 'some_module'
                                     (this will be visited first)
                                     then later, cv2.some_module.CONSTANT,
                                     node.value is Attribute(value=Name(id='cv2'), attr='some_module')
        """
        # Check if the base of the attribute access (e.g., 'cv' in 'cv.imread')
        # is an ast.Name node and matches one of the incorrect prefixes.
        if (
            isinstance(node.value, ast.Name)
            and node.value.id in self.incorrect_prefixes
        ):
            # original_prefix = node.value.id
            # Change the id of the ast.Name node to the correct prefix
            node.value.id = self.correct_prefix
            self.namespace_updated = True
        # Important: continue visiting child nodes.
        # This handles cases like cv.submodule.function, ensuring 'cv' is
        # replaced.
        return self.generic_visit(node)


def ensure_temp_dir(temp_output_dir):
    """Ensures the temporary directory for processed images exists."""
    os.makedirs(temp_output_dir, exist_ok=True)
    # Also ensure the directory is writable (basic check)
    if not os.access(temp_output_dir, os.W_OK):
        raise PermissionError(
            f"Temporary directory {temp_output_dir} is not writable.")


# (Default) For Linux/Unix system, signal mechanism could be used (high-efficiency).
# For Windows where signal support is invalid, this function will be
# executed in a separate process via multi-processing mechanism.


@timeout_decorator.timeout(EXEC_TIME_LIMIT, use_signals=True)
def _sandboxed_execution_target(
    code_to_execute,
    input_image_path,
    temp_output_dir,
    item_id,
    previous_execution_context=None,
):
    """
    Target function for sandboxed code execution in a subprocess.
    Results are put into return_dict.
    `previous_execution_context` is a dict {'globals': ..., 'locals': ...}
    """
    return_dict = {}

    # Prepare a restricted environment for exec()
    allowed_builtins = {
        "print": print,
        "len": len,
        "str": str,
        "int": int,
        "float": float,
        "list": list,
        "dict": dict,
        "tuple": tuple,
        "range": range,
        "round": round,
        "abs": abs,
        "min": min,
        "max": max,
        "sum": sum,
        "sorted": sorted,
        "any": any,
        "all": all,
        "zip": zip,
        "map": map,
        "filter": filter,
        "True": True,
        "False": False,
        "None": None,
        "isinstance": isinstance,
        "issubclass": issubclass,
        "Exception": Exception,
        "ValueError": ValueError,
        "TypeError": TypeError,
        "AttributeError": AttributeError,
        "IndexError": IndexError,
        "KeyError": KeyError,
        "NotImplementedError": NotImplementedError,
        "enumerate": enumerate,
        "pow": pow,
        "divmod": divmod,
        "bin": bin,
        "oct": oct,
        "hex": hex,
        "complex": complex,
        "__import__": __import__,
        "globals": globals,
        "locals": locals,
        "open": open,
        # Modules accessible via __import__ or provided in sandbox_globals
        "os": os,
        "shutil": shutil,
        "itertools": __import__("itertools"),
        "re": __import__("re"),
        "time": __import__("time"),
        "datetime": __import__("datetime"),
        "math": __import__("math"),
        "cmath": __import__("cmath"),
        "collections": __import__("collections"),
        "json": json,
        "PIL": __import__("PIL"),
        "random": random,
        "UnidentifiedImageError": UnidentifiedImageError,
    }

    sandbox_globals = {
        "__builtins__": allowed_builtins,
        "os": os,
        "random": random,
        "string": string,
        "math": math,
        "Image": Image,
        "UnidentifiedImageError": UnidentifiedImageError,
        "numpy": numpy,
        "np": numpy,  # TODO: pandayin: Check this? this seems like a duplicate
        "json": json,  # Make json directly available
        "re": re,  # Make re directly available
        # PIL is available via __import__ or Image class
    }

    if cv2:
        sandbox_globals["cv2"] = cv2

    sandbox_globals_always = sandbox_globals
    sandbox_locals = {
        "image_path": input_image_path,
        "temp_output_dir": temp_output_dir,
    }

    if previous_execution_context:
        # Restore picklable local variables from the previous step
        sandbox_locals.update(previous_execution_context.get("locals", {}))

        # Re-create imported modules from the previous step
        imports_to_recreate = previous_execution_context.get("globals", {})
        for alias, module_name in imports_to_recreate.items():
            try:
                # Re-import the module and add it to the current step's locals
                module_obj = importlib.import_module(module_name)
                sandbox_locals[alias] = module_obj
            except ImportError:
                print(
                    f"Warning: Could not re-import module '{module_name}' (as '{alias}') from previous step."
                )

    code_to_execute = align_first_line_to_second(code_to_execute)
    try:
        import autopep8
        dedented_code = textwrap.dedent(code_to_execute).strip()
        code_to_execute = dedented_code
        formatted_code = autopep8.fix_code(
            code_to_execute, options={"aggressive": 2}
        )
        # if formatted_code.strip() != code_to_execute.strip():
        #     print(f"INFO: Attempted to auto-format code for {item_id} using autopep8.")
        code_to_execute = formatted_code
    except Exception:
        pass

    # ... (AST transformations: ImagePathTransformer, CropCoordinateTransformer, OpenCVNamespaceTransformer) ...
    # (These transformations should operate on code_to_execute before exec)
    if input_image_path and isinstance(input_image_path, str):
        try:
            tree = ast.parse(code_to_execute)
            transformer = ImagePathTransformer(input_image_path)
            new_tree = transformer.visit(tree)
            if transformer.path_was_replaced:
                if hasattr(ast, "unparse"):
                    code_to_execute = ast.unparse(new_tree)
                else:
                    print(
                        "Warning: ast.unparse not available (requires Python 3.9+). "
                        "Code for image_path replacement not updated. Consider installing"
                        "'astor' for older Python versions or upgrading Python."
                    )
        except SyntaxError as e:
            print(
                f"Syntax error when parsing code for image_path replacement: {e}")
        except Exception as e:
            print(
                f"An error occurred during image_path AST transformation: {e}")

    actual_image_width, actual_image_height = None, None
    if input_image_path and os.path.isfile(input_image_path):
        try:
            with Image.open(input_image_path) as img_obj:
                actual_image_width, actual_image_height = img_obj.size
            if actual_image_width is None or actual_image_height is None:
                print(
                    f"WARNING: Could not determine dimensions for '{input_image_path}' using PIL."
                )
        except Exception as e:
            print(
                f"WARNING: Could not read image dimensions from '{input_image_path}' using PIL: {e}"
            )

    if actual_image_width is not None and actual_image_height is not None:
        if actual_image_width == 0 or actual_image_height == 0:
            print(
                f"WARNING: Input image '{input_image_path}' has zero width or height"
                f"({actual_image_width}x{actual_image_height}). Clamping might result in zero-size crops."
            )
        try:
            crop_ast_tree = ast.parse(code_to_execute)
            crop_transformer = CropCoordinateTransformer(
                actual_image_width, actual_image_height
            )
            new_crop_ast_tree = crop_transformer.visit(crop_ast_tree)
            if crop_transformer.coordinates_clamped:
                if hasattr(ast, "unparse"):
                    code_to_execute = ast.unparse(new_crop_ast_tree)
                else:
                    print(
                        "WARNING: ast.unparse not available (Python 3.9+)."
                        "Crop coordinate clamping not fully updated in code string."
                    )
        except SyntaxError:
            # print(f"ERROR: Syntax error when parsing code for crop coordinate clamping: {e}")
            pass
        except Exception as e:
            print(
                f"ERROR: An error occurred during crop coordinate AST transformation: {e}"
            )
            pass
    else:
        if input_image_path:
            # print(
            #     f"INFO: Skipping crop coordinate clamping because image dimensions"
            #     f"could not be obtained for '{input_image_path}'.")
            pass

    try:  # Wrap AST parsing for OpenCV transformer in try-except
        tree = ast.parse(code_to_execute)
        cv_transformer = OpenCVNamespaceTransformer(correct_prefix="cv2")
        new_tree = cv_transformer.visit(tree)

        if cv_transformer.namespace_updated:
            print("OpenCV namespace references were updated.")
            if hasattr(ast, "unparse"):
                code_to_execute = ast.unparse(new_tree)
                print("Code updated using ast.unparse.")
            else:
                try:
                    import astor

                    code_to_execute = astor.to_source(new_tree)
                    print("Code updated using astor.to_source.")
                except ImportError:
                    print(
                        "WARNING: ast.unparse not available (Python 3.9+), and astor not installed."
                    )
                    print(
                        "The AST was modified, but the code_to_execute string was not updated."
                    )
    except SyntaxError as e:
        print(
            f"ERROR: Syntax error when parsing code for OpenCV namespace transformation: {e}."
        )
    except Exception as e:
        print(
            f"ERROR: An error occurred during OpenCV namespace AST transformation: {e}"
        )

    captured_stdout = io.StringIO()
    processed_path_from_code = None
    processed_paths_list = []
    error_msg = None
    full_print_output = None

    try:
        with contextlib.redirect_stdout(captured_stdout):
            original_prefix = "/mnt/data/temp_processed_images/"
            replacement_target_path = temp_output_dir
            if original_prefix.endswith(
                    "/") and not temp_output_dir.endswith("/"):
                replacement_target_path = temp_output_dir + "/"

            # Create subdirectories if implied by paths in code
            quoted_path_pattern_str = (
                r"(['\"])(" + re.escape(original_prefix) + r"[^'\"]*)\1"
            )
            quoted_path_pattern = re.compile(quoted_path_pattern_str)
            unique_paths_found = set()
            for match in quoted_path_pattern.finditer(code_to_execute):
                unique_paths_found.add(match.group(2))
            for full_path_str in unique_paths_found:
                if full_path_str.startswith(original_prefix):
                    # relative_path_suffix = full_path_str[len(original_prefix):]
                    # print('full_path_str', unique_paths_found)
                    relative_path_suffix = full_path_str[
                        len(original_prefix):
                    ].replace(" ", "")
                    if relative_path_suffix:
                        path_directory_part = os.path.dirname(
                            relative_path_suffix)
                        if path_directory_part:
                            target_subdir_to_create = os.path.join(
                                temp_output_dir, path_directory_part
                            )
                            os.makedirs(target_subdir_to_create, exist_ok=True)

            code_to_execute = code_to_execute.replace(
                original_prefix, replacement_target_path
            )
            code_to_execute = re.sub(r":\.\d{1}f}", ":.8f}", code_to_execute)
            exec(code_to_execute, sandbox_globals, sandbox_locals)

        full_print_output = captured_stdout.getvalue().strip()

        if "processed_path" in sandbox_locals and (
            not previous_execution_context
            or "processed_path" not in previous_execution_context.get("locals", {})
        ):
            processed_path_from_code = sandbox_locals["processed_path"]
            if (
                isinstance(processed_path_from_code, str)
                and processed_path_from_code.startswith(temp_output_dir)
                and os.path.isfile(processed_path_from_code)
            ):
                processed_paths_list.append(processed_path_from_code)
            elif (
                isinstance(processed_path_from_code, str)
                and processed_path_from_code.startswith(temp_output_dir)
                and not os.path.exists(processed_path_from_code)
            ):
                error_msg = (
                    f"Sandbox for {item_id}: 'processed_path' variable set to "
                    f"'{processed_path_from_code}', but file does not exist."
                )
                processed_path_from_code = None
            elif processed_path_from_code is not None:
                error_msg = (
                    f"Sandbox for {item_id}: 'processed_path' variable was "
                    f"'{processed_path_from_code}', which is not a valid file path in {temp_output_dir}."
                )
                processed_path_from_code = None
        # print(full_print_output)
        if full_print_output:
            path_search_pattern = rf"({re.escape(temp_output_dir)}[^\s\'\"]+\.(?:jpg|jpeg|png|bmp|gif|tiff))"
            possible_image_path_list = get_image_paths(temp_output_dir)
            possible_error_msg = ""
            num_parse_images = 0
            for match in re.finditer(path_search_pattern, full_print_output):
                potential_path_from_print = match.group(1)
                num_parse_images += 1
                if os.path.isfile(potential_path_from_print):
                    if potential_path_from_print not in processed_paths_list:
                        processed_paths_list.append(potential_path_from_print)
                elif not error_msg:
                    possible_error_msg = (
                        f"Sandbox for {item_id}: Path '{potential_path_from_print}' "
                        "found in print, but file does not exist or is not a file."
                    )
            if len(processed_paths_list) == 0:
                if num_parse_images == len(possible_image_path_list):
                    processed_paths_list = possible_image_path_list
                else:
                    path_search_pattern = (
                        r"([^\s\'\"]+\.(?:jpg|jpeg|png|bmp|gif|tiff))"
                    )
                    list_of_all_matches = re.findall(
                        path_search_pattern, full_print_output
                    )
                    if len(list_of_all_matches) == len(
                            possible_image_path_list):
                        processed_paths_list = possible_image_path_list
                    else:
                        error_msg = possible_error_msg

            if not processed_paths_list and temp_output_dir not in code_to_execute:
                try:
                    processed_paths_list.append(full_print_output)
                except ValueError:
                    pass

    except ImportError as e:
        error_msg = (
            f"Sandbox for {item_id}: Code execution failed due to ImportError. "
            f"Ensure all required modules are available and correctly named: {e}"
        )
        if "cv2" in str(e).lower() and not cv2:
            error_msg += (
                f"(Note: cv2 was not available in the sandbox host environment): {e}"
            )
    except MemoryError as e:
        error_msg = (
            f"Sandbox for {item_id}: Code execution failed due to MemoryError. "
            f"The operation likely consumed too much memory: {e}"
        )
    except SyntaxError as e:  # Catch syntax errors from exec itself
        error_msg = (
            f"Sandbox for {item_id}: Code execution failed due to SyntaxError: {e}"
        )
    except Exception as e:
        error_msg = f"Sandbox for {item_id}: Code execution failed: {e}"

    return_dict["processed_paths_list"] = processed_paths_list
    return_dict["print_output"] = full_print_output
    return_dict["error"] = error_msg

    if full_print_output is not None and not processed_paths_list:
        error_msg = f"Sandbox for {item_id}: Path/result output error, unable to match save path"

    picklable_variables = {}
    imports_to_persist = {}

    for name, value in sandbox_locals.items():
        if name == "__builtins__":
            continue
        if name in sandbox_globals_always.keys():
            continue
        if isinstance(value, types.ModuleType):
            imports_to_persist[name] = value.__name__
        else:
            try:
                # pickle.dumps(value)
                picklable_variables[name] = value
            except (pickle.PicklingError, TypeError):
                print(
                    f"Warning: Var '{name}' of type {type(value).__name__} is not picklable.",
                    file=sys.stderr,
                )

    # Store the final state of globals and locals
    return_dict["execution_context"] = {
        "globals": imports_to_persist,
        "locals": picklable_variables,
    }
    return return_dict


def execute_code_in_sandbox(
    code_to_execute,
    input_image_path,
    item_id="N/A",
    temp_output_dir=None,
    previous_execution_context=None,
):
    """
    Executes Python code in a restricted sandbox using timeout_decorator.

    Args:
        code_to_execute (str): The Python code string to execute.
        input_image_path (str): Path to the user image.
        item_id (str): Identifier for logging.
        temp_output_dir (str): Directory for processed images.
        previous_execution_context (dict, optional): Context from a previous execution.

    Returns:
        tuple: (processed_file_paths_list, captured_print_output,
                error_message_or_none, current_execution_context)
    """
    if temp_output_dir is None:
        temp_output_dir = TEMP_PROCESSED_IMAGES_DIR
    if check_dangerous_code(code_to_execute):
        return (
            [],
            "",
            (
                f"Sandbox for {item_id}: Code contains potentially dangerous system operations "
                "such as remove. Execution denied.",
            ),
            None,
        )

    ensure_temp_dir(temp_output_dir)
    with ReadOnlyPath(input_image_path):
        try:
            # Directly call the decorated function
            result_dict = _sandboxed_execution_target(
                code_to_execute,
                input_image_path,
                temp_output_dir,
                item_id,
                previous_execution_context,
            )

            # Unpack results from the returned dictionary
            processed_paths_list = result_dict.get("processed_paths_list", [])
            full_print_output = result_dict.get("print_output", "")
            error_msg = result_dict.get("error", None)
            current_execution_context = result_dict.get(
                "execution_context", {"globals": {}, "locals": {}}
            )

        except timeout_decorator.TimeoutError:
            # Handle the timeout gracefully
            error_msg = f"Sandbox for {item_id}: Execution timed out after {EXEC_TIME_LIMIT} seconds."
            print(error_msg)
            processed_paths_list = []
            full_print_output = ""
            current_execution_context = None

    return processed_paths_list, full_print_output, error_msg, current_execution_context
