import json
from statistics import mean
import time

import Levenshtein
import dspy
from pydantic.json import pydantic_encoder

class SelfConsistency(dspy.Module):
    def __init__(self, signature, generations_number=5, cooldown_between_generations_sec=0):
        super().__init__()

        self.predict = dspy.Predict(signature=signature)
        self.generations_number = generations_number
        self.cooldown = cooldown_between_generations_sec

    def forward(self, _dspy_context_lm=None,**kwargs):
        ctx_args = {'lm':_dspy_context_lm.copy(cache=False) if _dspy_context_lm is not None else dspy.settings.lm.copy(cache=False) }
        if ctx_args['lm'].kwargs['temperature'] == 0:
            print("Warning: temperature for self-consistency is set to 0.0, there'll be no variation in the generations.")

        generations = []
        with dspy.context(**ctx_args):
            for _ in range(self.generations_number):
                generations.append(self.predict(**kwargs))
                time.sleep(self.cooldown)
        return self.most_similar_object(generations)

    @classmethod
    def most_similar_object(cls, generations): #object with least mean levenshtein distance from the others (most similar to the others compared to any other object).
                
        def normalized_levenshtein_distance(str1, str2):
            distance = Levenshtein.distance(str1, str2)
            max_len = max(len(str1), len(str2))
            return distance / max_len if max_len > 0 else 0

        # Calculate the average distance of each object to all other objects
        average_distances = {}

        generation_jsons = []
        for g in generations:
            generation_jsons.append(json.dumps(g.toDict(), default=pydantic_encoder))
        for i, obj1 in enumerate(generation_jsons):
            distances = []
            
            for j, obj2 in enumerate(generation_jsons):
                if i != j:
                    distance = normalized_levenshtein_distance(obj1, obj2)
                    distances.append(distance)

            # Use JSON string as key and store the average distance for the object
            average_distances[i] = mean(distances) if distances else float('inf')
            
        # print("Average Distances:")
        # for ind, avg_dist in average_distances.items():
        #     print(f"{generation_jsons[ind]}: {round(avg_dist, 4)}")

        return generations[min(average_distances, key=average_distances.get)]