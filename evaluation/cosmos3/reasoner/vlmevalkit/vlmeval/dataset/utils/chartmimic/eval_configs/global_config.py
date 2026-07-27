import subprocess

texts = []
images = []
markers = []


def reset_texts():
    texts.clear()


def add_text(text):
    texts.append(text)


def get_raw_texts():
    return [item[2] for item in texts]


def get_texts():
    return texts


def reset_images():
    images.clear()


def add_image(image):
    images.append(image)


def get_images():
    return images


def reset_markers():
    markers.clear()


def add_marker(marker):
    markers.append(marker)


def get_markers():
    return markers


def run_script_safe(script_path):
    try:
        subprocess.run(
            ["python3", script_path],
            check=True,
            capture_output=True,
            text=True
        )
        return True  # success
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to run {script_path}")
        print(f"[Return Code]: {e.returncode}")
        print(f"[Stdout]:\n{e.stdout}")
        print(f"[Stderr]:\n{e.stderr}")
        return False  # failed
