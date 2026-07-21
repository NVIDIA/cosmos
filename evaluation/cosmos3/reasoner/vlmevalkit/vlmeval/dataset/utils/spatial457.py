SUPERCLRVER_sub_shape = {
    "car": ["suv", "wagon", "minivan", "sedan", "truck", "addi", "car"],
    "bus": ["articulated", "regular", "double", "school", "bus"],
    "motorbike": ["chopper", "dirtbike", "scooter", "cruiser", "motorbike"],
    "aeroplane": ["jet", "fighter", "biplane", "airliner", "aeroplane"],
    "bicycle": ["road", "utility", "mountain", "tandem", "bicycle"],
}

inverse_shape = {}
for key, value in SUPERCLRVER_sub_shape.items():
    for v in value:
        inverse_shape[v] = key


class Spatial457_utils:
    def __init__(self):

        return

    def get_random_answer(self, gt):
        import random

        all_attributes = {
            "size": ["small", "large"],
            "shape": [
                "airliner",
                "dirtbike",
                "road bike",
                "tandem bike",
                "suv",
                "wagon",
                "scooter",
                "mountain bike",
                "minivan",
                "sedan",
                "school bus",
                "fighter",
                "chopper",
                "double bus",
                "truck",
                "articulated bus",
                "cruiser",
                "jet",
                "utility bike",
                "regular bus",
                "biplane",
            ],
            "color": [
                "gray",
                "blue",
                "purple",
                "brown",
                "green",
                "cyan",
                "red",
                "yellow",
            ],
            "direction": ["left", "right", "front", "back"],
        }

        gt = gt.lower()
        if gt in ["yes", "no"]:
            return random.choice(["yes", "no"])
        if gt in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
            return str(random.randint(0, 9))
        for key, value in all_attributes.items():
            if gt in value:
                return random.choice(value)

    def all_answers(self):
        all_attributes = {
            "size": ["small", "large"],
            "shape": [
                "airliner",
                "dirtbike",
                "road bike",
                "tandem bike",
                "suv",
                "wagon",
                "scooter",
                "mountain bike",
                "minivan",
                "sedan",
                "school bus",
                "fighter",
                "chopper",
                "double bus",
                "truck",
                "articulated bus",
                "cruiser",
                "jet",
                "utility bike",
                "regular bus",
                "biplane",
            ],
            "color": [
                "gray",
                "blue",
                "purple",
                "brown",
                "green",
                "cyan",
                "red",
                "yellow",
            ],
            "direction": ["left", "right", "front", "back"],
        }

        all_answers = ""
        for key, value in all_attributes.items():
            captical_value = [x.capitalize() for x in value]
            all_answers += ", ".join(captical_value) + ", "
        return all_answers.strip(", ")

    def is_correct(self, answer, predict):
        text2num = {
            "zero": "0",
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9",
            "ten": "10",
        }
        predict = str(predict)
        answer = str(answer)

        if predict.lower() == "none":
            predict = "no"

        if predict.lower() == answer.lower():
            return True
        if predict == "0" and answer == "No":
            return True
        if predict.lower() in text2num and text2num[predict.lower()] == answer:
            return True
        if answer.lower() == "yes" and predict in [
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
        ]:
            return True

        if self.category_correct(predict, answer):
            return True
        return False

    def category_correct(self, answer, gt_answer):
        answer = str(answer).lower().split(" ")[0]
        gt_answer = str(gt_answer).lower().split(" ")[0]

        if (
            answer in inverse_shape
            and gt_answer in inverse_shape
            and inverse_shape[answer] == inverse_shape[gt_answer]
        ):
            return True

        return False
