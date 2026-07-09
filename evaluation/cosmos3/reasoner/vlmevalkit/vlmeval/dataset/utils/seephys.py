import re
from collections import defaultdict

from sympy import Eq, latex, simplify
from sympy.parsing.latex import parse_latex

from vlmeval.smp import load

FAIL_MSG = 'Failed to obtain answer via API.'

prompt_scoring = r"""
You are a physics professor, please determine if the Standard answer and Model Answer are equivalent. Note that the significant figures in the answer must meet the requirements. Your judgment should be 0 (non-equivalent) or 1 (equivalent).

[Question]: A force of 20 N acts on an object of mass 5 kg. What is the acceleration of the object?
[Standard Answer]: 4 m/s²
[Model Answer] : 4
Judgement: 1

[Question]: A projectile is launched at an angle $\\theta$ with initial velocity $v_0$. What is its time of flight before returning to the same height, assuming negligible air resistance and gravitational acceleration $g$?
[Standard Answer]: $$ t = \\frac{{2 v_0 \\sin(\\theta)}}{{g}} $$
[Model Answer] : Extracted Answer: $$ t = \\frac{{2 v_0 \\cos(\\frac{\\pi}{2} - \\theta)}}{{g}} $$
Judgement: 1

[Question]: The position of a particle is given by $x(t) = 3t^2 - 2t + 5$ meters. What is its instantaneous velocity at $t=2$ seconds?
[Standard Answer]: 10 m/s
[Model Answer] : Velocity $v(t) = dx/dt = 6t - 2$. At $t=2s$, $v(2) = 6(2) - 2 = 12 - 2 = 10$. So the velocity is 10 m/s.
Judgement: 1

[Question]: A car travels North at 20 m/s. It then turns and travels East at 20 m/s. What is the magnitude of its change in velocity?
[Standard Answer]: Approximately 28.3 m/s
[Model Answer] : The change in velocity is 0 m/s because the speed is the same.
Judgement: 0

[Question]: An object is thrown horizontally from a height of 20m with an initial speed of 10 m/s. Calculate: (a) the time it takes to hit the ground ($t_g$), and (b) the horizontal distance ($d_x$) it travels before hitting the ground. (Use g = 10 m/s²)
[Standard Answer]: (a) $t_g = 2$ s, (b) $d_x = 20$ m
[Model Answer] : (a) The time to hit the ground $t_g$ is 2 s. (b) The horizontal distance $d_x$ is 10 m.
Judgement: 0

[Question]: An engine performs $1.2 \\times 10^5$ J of work in 2 minutes. What is its average power output in watts?
[Standard Answer]: 1 kW
[Model Answer] : Power = Work / Time = $1.2 \\times 10^5$ J / (2 min * 60 s/min) = $1.2 \\times 10^5$ J / 120 s = 1000 W.
Judgement: 1

[Question]: A resistor has a voltage of 10V across it and a current of 2A flowing through it. What is its resistance and power dissipation?
[Standard Answer]: Resistance R = 5 Ohms , Power P = 20 Watts.
[Model Answer] : The resistance is $R = V/I = 10V / 2A = 5 \Omega$. The power dissipated is $P = VI = 10V \\times 2A = 20W$.
Judgement: 1

[Question]: The displacement of an object in Simple Harmonic Motion (SHM) is given by $x(t) = A \sin(\omega t)$. Determine the equation for its acceleration, $a(t)$.
[Standard Answer]: $$ a(t) = -A\omega^2 \sin(\omega t) $$
[Model Answer] : The acceleration is the second derivative of displacement. $v(t) = A\omega \cos(\omega t)$. $a(t) = A\omega^2 \cos\left(\omega t + \\frac{\pi}{2}\\right)$.
Judgement: 1

[Question]: 给出相对论性粒子总能量 $E$ 的速度展开式（到 $v^4/c^4$ 项）。
[Standard Answer]: $E = mc^2 \left(1 + \frac{v^2}{2c^2} + \frac{3v^4}{8c^4} + \mathcal{O}(v^6/c^6)\right)$
[Model Answer]: $E = \gamma m c^2 = \frac{mc^2}{\sqrt{1 - v^2/c^2}} \approx mc^2 + \frac{1}{2}mv^2 + \frac{3}{8} \frac{mv^4}{c^2}$
Judgement: 1

[Question]: 计算粒子能量 $E$ 穿过势垒 $V_0$ ($E < V_0$) 的透射系数 $T$。
[Standard Answer]: $\ln T \approx \ln 16 + \ln\left(\frac{E}{V_0}\right) + \ln\left(1 - \frac{E}{V_0}\right) - \frac{2d}{\hbar} \sqrt{2m(V_0 - E)}$
[Model Answer]: $T \approx 16 \frac{E}{V_0} \left(1 - \frac{E}{V_0}\right) e^{-2d\sqrt{2m(V_0 - E)}/\hbar}$
Judgement: 1

[Question]: The position of a particle is given by $x(t) = (2t^3 - 3t)$ meters. What is its acceleration at $t=1$ second? The final answer should retain 3 significant figures.
[Standard Answer]: 12.0 m/s²
[Model Answer] : $v(t) = 6t^2 - 3$. $a(t) = 12.1t$. At $t=1s$, $a(1) = 12.1 \\text{ m/s}^2$.
Judgement: 0
---
Now please provide your judgement (0 or 1), DONNOT output explanation:
""" # noqa


def get_example():
    example_1 = """
Question: What is the net force acting on a 5 kg object accelerating at 3 m/s² to the right?\n
Model response: Using F = ma, the net force is 15 N to the right.\n
Extracted answer: the net force is 15 N to the right.
""" # noqa

    example_2 = """
Question: Calculate the charge of an electron. (Unit: C)\n
Model response: The elementary charge of an electron is approximately -1.602 × 10⁻¹⁹ coulombs.\n
Extracted answer: -1.602 × 10⁻¹⁹ C
""" # noqa

    example_3 = """
Question: How much money does Luca need to buy a sour apple candy and a butter-scotch candy? (Unit: $)\n
Model response: Luca needs $1.45 to buy a sour apple candy and a butterscotch candy.\n
Extracted answer: $1.45
""" # noqa

    example_4 = """
Question: Between which frequencies does human hearing typically range? \n
Model response: Human hearing ranges between 20 Hz and 20,000 Hz.\n
Extracted answer: [20 Hz, 20000 Hz]
""" # noqa

    example_5 = """
Question: List the wavelengths of visible light colors.\n
Model response: Visible light ranges from:\n
- Red: ~700 nm\n
- Green: ~550 nm\n
- Blue: ~450 nanometre\n
Extracted answer: Red: 700 nm; Green: 550 nm; Blue: 450 nanometre.
""" # noqa
    return [example_1, example_2, example_3, example_4, example_5]


def build_extract_prompt(line):
    task_description = """
Please read the following example.
Then extract the answer from the model response and type it at the end of the prompt.\n # noqa
"""
    question = "Please answer this question in the image." if str(line['question']) == 'nan' else line['question'] # noqa

    prediction = extract_by_rule(line)
    prompt = task_description
    examples = get_example()
    for example in examples:
        prompt += example + '\n'
    prompt += question + '\n'
    prompt += 'Model response: ' + prediction
    prompt += 'Extracted answer:'
    return prompt


def list_to_dict(lst):
    return {chr(65 + i): val for i, val in enumerate(lst)}


def extract_by_rule(line):
    response = line['prediction']
    try:
        pattern = r'<answer>\s*(\d+)\s*</answer>'
        match = re.search(pattern, response)
        if match:
            response = match.group(1)
            return response

    except Exception:
        pass
    try:
        pattern = r"the final answer is: (.+?)\."
        match = re.search(pattern, response)
        if match:
            response = match.group(1)
            return response
    except Exception:
        pass
    try:
        pattern = r"The answer is: (.+?)\."
        match = re.search(pattern, response)
        if match:
            response = match.group(1)
            return response
    except Exception:
        pass

    try:
        response = int(response)
        return str(response)
    except Exception:
        pass
    try:
        response = float(response)
        return str(response)
    except Exception:
        pass
    return str(response)


def quick_compare(response, answer, tol=1e-6):
    if response is None or answer is None:
        return False

    # 场景1：比较数值（如 "3.1415" vs "\pi"）
    if response.is_Number and answer.is_Number:
        return abs(float(response - answer)) < tol
    # 场景2：比较等式（如 "x = y" vs "y = x"）
    if isinstance(response, Eq) and isinstance(answer, Eq):
        return simplify(response.lhs - response.rhs) == simplify(answer.lhs - answer.rhs)

    # 场景3：比较表达式（如 "\frac{x}{y}" vs "x/y"）
    return simplify(response - answer) == 0


def post_check(line, prefetch=False):
    # prefetch: return extracted response
    ans = line['answer']
    try:
        res = extract_by_rule(line)
    except ValueError:
        return False

    if str(res) == str(ans):
        return str(res) if prefetch else True

    try:
        parsed_res = parse_latex(res)
        parsed_ans = parse_latex(ans)
        if quick_compare(parsed_res, parsed_ans):
            return latex(parsed_res) if prefetch else True
    except Exception:
        return False
    return False


def extract(model, line):
    log = ''
    retry = 5
    if post_check(line, prefetch=False):
        res = post_check(line, prefetch=True)
        return dict(log='Prefetch succeed', extract=res, score=1)
    else:
        prompt = build_extract_prompt(line)
        for i in range(retry):
            prediction = line['prediction']
            res = model.generate(prompt, temperature=i * 0.5)  # extract
            if not res or FAIL_MSG in res:
                log += f'Try {i}: output is {prediction}, failed to parse.\n'
            else:
                log += 'Succeed'
                score = score_func(model, res, line['question'], line['answer'])
                if score is None:
                    log += '\nScore failed'
                    return dict(log=log, extract=res, score=-1)
                return dict(log=log, extract=res, score=score)
    log += 'All 5 retries failed.\n'
    return dict(log=log, extract='', score=-1)


def score_func(model, response, query, gt):
    if not response:
        return 0
    try:
        full_prompt = (
            prompt_scoring.strip()
            + f"\n[Question]: {query}\n[Standard Answer]: {gt}\n[Model Answer]: {response}\nJudgement: "
        )
        try_n = 0
        while try_n < 5:
            score = model.generate(full_prompt, temperature=try_n * 0.3)
            if 'Judgement: ' in score:
                score = score.split('Judgement: ')[-1]
            elif 'Judgement:' in score:
                score = score.split('Judgement:')[-1]
            elif 'judgement: ' in score:
                score = score.split('judgement: ')[-1]
            elif 'judgement:' in score:
                score = score.split('judgement:')[-1]
            try:
                if int(score) == 0 or int(score) == 1:
                    return int(score)
            except Exception:
                continue
    except Exception as e:
        print("score_func Error!")
        print(e)
        return None


def eval_acc(result_file):
    data = load(result_file)
    keys = ['level', 'subject', 'language', 'source', 'vision_relevance', 'img_category', 'sig_figs']
    keys = [k for k in keys if k in data]
    tot = {k: defaultdict(lambda: 0) for k in keys}
    fetch = {k: defaultdict(lambda: 0) for k in keys}
    hit = {k: defaultdict(lambda: 0) for k in keys}
    tot['Overall'] = 0
    fetch['Overall'] = 0
    hit['Overall'] = 0

    lt = len(data)
    for i in range(lt):
        item = data.iloc[i]
        tot['Overall'] += 1
        for k in keys:
            value = str(item[k])
            tot[k][value] += 1

        if 'Prefetch succeed' in item['log']:
            fetch['Overall'] += 1
            for k in keys:
                value = str(item[k])
                fetch[k][value] += 1

        if post_check(item, prefetch=False):
            hit['Overall'] += 1
            for k in keys:
                value = str(item[k])
                hit[k][value] += 1
        elif item['score'] == 1:
            hit['Overall'] += 1
            for k in keys:
                value = str(item[k])
                hit[k][value] += 1

    res = {k: defaultdict(lambda: 0) for k in keys}
    res['acc'] = 0
    res['prefetch_rate'] = 0

    res['acc'] = hit['Overall'] / tot['Overall'] * 100 if tot['Overall'] > 0 else 0
    res['prefetch_rate'] = fetch['Overall'] / tot['Overall'] * 100 if tot['Overall'] > 0 else 0

    def calculate_accuracy(hit_dict, tot_dict, res_dict, category):
        for key in tot_dict[category]:
            total = tot_dict[category][key]
            hits = hit_dict[category][key]
            res_dict[category][key] = hits / total * 100 if total > 0 else 0

    for category in keys:
        calculate_accuracy(hit, tot, res, category)
    res_dict = {
        'Overall': {
            'Accuracy (%)': res['acc'], 'PrefetchRate (%)': res['prefetch_rate']
        }, **{cat: dict(res[cat]) for cat in keys}
    }
    return res_dict
