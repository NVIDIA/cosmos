import re

PARAM_LIST_MATCHER = re.compile(r"((?:\?\S+\s*)+)(?:-\s+([^\?$]+)\s*)?")
PARAM_NAME_MATCHER = re.compile(r"\?([^\s\?\)]+)\s*")


##### Parsing functions and parentheses matching
def parse_pddl_param_list(s):
    s = s.strip()
    assert s[0] == "(" and s[-1] == ")"
    s = s[1:-1]
    param_type_dict = {}
    for params, p_type in PARAM_LIST_MATCHER.findall(s):
        for p in PARAM_NAME_MATCHER.findall(params):
            p_type = p_type.strip()
            if p_type.startswith("("):
                p_type = p_type[1:-1].strip()
                assert "either"
                param_type_dict[p] = re.split(r"\s+", p_type)[1:]
            else:
                param_type_dict[p] = p_type
    return s.split("?")[0].strip(), param_type_dict


def parse_outer_inner_str(s, str_ender, inner_starter, inner_ender):
    inner_count = 0
    start_id = 0
    matched_str = []
    for i, c in enumerate(s):
        if inner_count == 0 and c == str_ender:
            return s[: i + 1], matched_str, i + 1
        elif c == inner_starter:
            if inner_count == 0:
                start_id = i
            inner_count += 1
        elif c == inner_ender:
            inner_count -= 1
            if inner_count == 0:
                matched_str.append(s[start_id : i + 1])
    return s, matched_str, len(s)


def parse_pddl_attr_from_string(
    s,
    attr_starter="(:",
    attr_ender=")",
    inner_starter="(",
    inner_ender=")",
    overlap=False,
):
    s_attr = s.split(attr_starter)
    if len(s_attr) == 1:
        return "", []
    elif len(s_attr) == 2:
        outer_str, inner_str, _ = parse_outer_inner_str(
            s_attr[1], attr_ender, inner_starter, inner_ender
        )
        return attr_starter + outer_str, inner_str
    else:
        matched_dict = {}
        outer_list = []
        if not overlap:
            while len(s.split(attr_starter)) > 1:
                s = s.split(attr_starter, 1)[1]
                name = re.split(r"\s+", s.strip())[0]
                outer_str, inner_str, end_point = parse_outer_inner_str(
                    s, attr_ender, inner_starter, inner_ender
                )
                outer_list.append(attr_starter + outer_str)
                matched_dict[name] = inner_str
                s = s[end_point:]
        else:
            for seg in s_attr[1:]:
                name = re.split(r"\s+", seg.strip())[0]
                outer_str, inner_str, _ = parse_outer_inner_str(
                    seg, attr_ender, inner_starter, inner_ender
                )
                outer_list.append(attr_starter + outer_str)
                matched_dict[name] = inner_str
        return outer_list, matched_dict


def remove_type_in_cnf(s):
    s_split_type = s.split(" - ")
    if len(s_split_type) > 1:
        for i in range(1, len(s_split_type)):
            if len(s_split_type[i].strip().split(")")[0].split()) == 1:
                s_split_type[i] = ")" + s_split_type[i].strip().split(")", 1)[1]
            else:
                s_split_type[i] = " " + s_split_type[i].strip().split(" ", 1)[1]
        return "".join(s_split_type).strip()
    else:
        return s


def split_cnf_by_parentheses(s):
    assert s.startswith("(and")
    matches = set()
    p_count = 0
    clause_start_id = 0
    for i in range(len(s)):
        if s[i] == "(":
            p_count += 1
            if p_count == 2:
                clause_start_id = i
        elif s[i] == ")":
            p_count -= 1
            if p_count == 0:
                break
            elif p_count == 1:
                matches.add(remove_type_in_cnf(s[clause_start_id : i + 1]))
    return matches


##### End of parsing functions


####### Domain (the env for each planning task)
class Domain:
    def __init__(self, name, domain_pddl):
        # self.name = name

        # Domain files
        self.domain_pddl = domain_pddl
        self.action_name, self.action_params, self.action_params_dict = (
            self.get_domain_action()
        )
        self.gt_cond_dict = self.parse_gt_pre_post_cond()

    def get_domain_action(self):
        action_pddl_str_list, all_actions = parse_pddl_attr_from_string(
            self.domain_pddl, attr_starter="(:action"
        )
        action_name, action_params, action_params_dict = [], [], []
        for action_pddl_str, (name, action_attr) in zip(
            action_pddl_str_list, all_actions.items()
        ):
            assert len(action_attr) == 3
            param_str, pre_cond_str, post_cond_str = action_attr
            action_name.append(name)
            action_params.append(param_str)
            action_params_dict.append(parse_pddl_param_list(param_str)[1])
        return action_name, action_params, action_params_dict

    def parse_gt_pre_post_cond(self):
        cond_dict = {}
        for a in self.action_name:
            act_str = self.domain_pddl.split(f"(:action {a}")[1]
            for postfix in ["pre", "post"]:
                split_tag = ":precondition" if postfix == "pre" else ":effect"
                cond_str = act_str.split(split_tag)[1].strip()
                if cond_str.startswith("(and"):
                    cond_dict[f"{a}_{postfix}"] = split_cnf_by_parentheses(cond_str)
                else:
                    cond_dict[f"{a}_{postfix}"] = {cond_str.split(")")[0].strip() + ")"}
                cond_dict[f"{a}_{postfix}"] = sorted(
                    list(cond_dict[f"{a}_{postfix}"]),
                    key=lambda x: 0 if x.startswith("(not ") else 1,
                )
        return cond_dict


##### Transition functions
def construct_param_to_obj(domain, action):
    action = action[1:-1]
    a_name = action.split(" ")[0].strip()
    objs = action.split(" ")[1:]
    a_index = domain.action_name.index(a_name)
    assert len(objs) == len(domain.action_params_dict[a_index])
    return {p: obj for p, obj in zip(domain.action_params_dict[a_index], objs)}, a_name


def state_transition(current_state, effects, param_to_obj):
    for obj_cond in effects:
        for param in param_to_obj:
            obj_cond = re.sub(
                r"\?{}(?=[^\w-])".format(param), param_to_obj[param], obj_cond
            )
        _, reversed_cond = parse_pddl_attr_from_string(obj_cond, attr_starter="(not ")
        if reversed_cond:
            assert len(reversed_cond) == 1
            if reversed_cond[0] in current_state:
                current_state.remove(reversed_cond[0])
        elif obj_cond.strip() not in current_state:
            current_state.append(obj_cond)
    return current_state


def check_pre_conds_satisfy(current_state, pre_conds, param_to_obj):
    for obj_cond in pre_conds:
        for param in param_to_obj:
            obj_cond = re.sub(
                r"\?{}(?=[^\w-])".format(param), param_to_obj[param], obj_cond
            )
        if (obj_cond.startswith("(not ") and obj_cond in current_state) or (
            not obj_cond.startswith("(not ") and obj_cond not in current_state
        ):
            return False
    return True


##### End of transition functions


class SymbolicPlanningMetricTest:
    """An example metric for symbolic planning tasks"""

    @classmethod
    def match(cls, response, eval_context, task_info=None):
        ## Initialize domain
        # task_name = task_info["task_name"]
        domain_pddl = eval_context["domain_pddl"]
        domain = Domain(" ", domain_pddl)

        ## Parse trajectory, setup initial and goal state
        # response = eval_context["gt_plan"]  # for debug
        match response:
            case str():
                candidates = response.split("\n")
            case tuple() | list():
                candidates = list(response)
            case _:
                raise ValueError(
                    f"`response` has unsupported type: {type(response)=}, {response=}"
                )
        cand_traj = [cand_a.strip() for cand_a in candidates if cand_a.startswith("(")]
        try:
            task_pddl = eval_context["task_pddl"]
            cur_state = parse_pddl_attr_from_string(task_pddl, attr_starter="(:init")[1]
            goal_state = parse_pddl_attr_from_string(task_pddl, attr_starter="(and")[1]
        except IndexError:
            score = 0
            return score

        score = 1
        try:
            ## State transitions and check if satisfy the preconditions
            for cand_a in cand_traj:
                param_to_obj, a_name = construct_param_to_obj(domain, cand_a)
                if not check_pre_conds_satisfy(
                    cur_state, domain.gt_cond_dict[f"{a_name}_pre"], param_to_obj
                ):
                    print(f"precondition of the action {cand_a} is not satisfied!")
                    score = 0
                    break
                cur_state = state_transition(
                    cur_state, domain.gt_cond_dict[f"{a_name}_post"], param_to_obj
                )

            ## Check if goal conditions are reached in the final state
            if score == 1:
                for g_state in goal_state:
                    if (g_state.startswith("(not ") and g_state in cur_state) or (
                        not g_state.startswith("(not ") and g_state not in cur_state
                    ):
                        print(f"goal state {g_state} is not reached!")
                        score = 0
                        break
        except ValueError:
            # grammar error in execution
            score = 0
        except AssertionError:
            # assertion error in functions
            score = 0

        return score
