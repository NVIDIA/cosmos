from enum import Enum


class AggregationType(Enum):
    MEAN = 0

    @classmethod
    def from_string(cls, s):
        return cls.MEAN

    def aggregate(self, field_scores, field_weights):
        if not field_scores:
            return 0.0

        total_score = 0.0
        total_weight = 0.0

        for field, score in field_scores.items():
            weight = field_weights.get(field, 1.0)
            try:
                total_score += score * weight
            except:
                total_score += score[0] * weight
            total_weight += weight

        return total_score / total_weight if total_weight > 0 else 0.0
