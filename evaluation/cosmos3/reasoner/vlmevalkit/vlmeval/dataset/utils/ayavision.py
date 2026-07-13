FAIL_MSG = "Failed to obtain answer via API."


def build_prompt_ayavision(line):
    question = line["question"]
    prediction = str(line["prediction"])
    answer = str(line["answer"])

    tmpl = (
        "You are an expert evaluator. Your task is to determine if the predicted answer "
        "is a correct response to the given question, using the ground truth answer as a reference. "
        "The predicted answer does not need to be a verbatim match of the ground truth, "
        "but it must be semantically equivalent and accurately answer the question.\n"
        "Respond with '[[CORRECT]]' if the prediction is correct, and '[[WRONG]]' if it is incorrect. "
        "Do not provide any explanation.\n\n"
        "Question: {question}\n"
        "Ground Truth Answer: {answer}\n"
        "Predicted Answer: {prediction}\n\n"
        "Is the prediction correct? "
    )
    return tmpl.format(question=question, answer=answer, prediction=prediction)


def AyaVision_auxeval(model, line):
    prompt = build_prompt_ayavision(line)
    log = ""
    retry = 5

    for i in range(retry):
        res = model.generate(prompt, temperature=i * 0.5)

        if FAIL_MSG in res:
            log += f"Try {i}: output is {res}, failed to parse.\\n"
        elif "[[CORRECT]]" in res:
            log += "Succeed"
            hit = 1
            return dict(log=log, res=res, hit=hit)
        elif "[[WRONG]]" in res:
            log += "Succeed"
            hit = 0
            return dict(log=log, res=res, hit=hit)
        else:
            log += f"Try {i}: output is {res}, failed to parse.\\n"

    log += "All 5 retries failed.\\n"
    return dict(log=log, res="", hit=0)
