from copy import deepcopy
from typing import List
import dspy
import pydantic

class IterativeRefiner(dspy.Module):
    def __init__(self, generate_signature):
        super().__init__()

        self.generate = dspy.Predict(generate_signature)
        self._set_critique_signature(generate_signature)
        self._set_refine_signature(generate_signature)

    def forward(self, parameters, refine_iterations=3):
        target_output_fields = [field_name for field_name, field_value in self.generate.signature.model_fields.items() if field_value.json_schema_extra['__dspy_field_type']=='output' ]
        generate_results = dict(self.generate(**parameters))
        # print(f"Results before refinement: {generate_results}")
        # print(f"-__-"*10)
        for i in range(refine_iterations):
            # print(f"Refine iteration #{i}")
            # print(f"-__-"*10)

            critique_results = self.critique(target_output_fields =target_output_fields, instruction_prompt=self.generate.signature.__doc__,  **{ **parameters, **generate_results})
            # print(f"Critique: {dict(critique_results)}")
            # print(f"-__-"*10)
            refine_results = self.refine(target_output_fields =target_output_fields, instruction_prompt=self.generate.signature.__doc__,  **{ **parameters, **generate_results, **{"critique":critique_results.critique, "feedback":critique_results.feedback}})
            for f_name, f_value in dict(refine_results).items():
                if f_name.startswith('refined_'):
                    generate_results.update({
                        f_name.replace('refined_', ''):f_value
                    })
            # print(f"Refined Results #{i}: {generate_results}")
            # print(f"-__-"*10)
        return dspy.Example(**generate_results).with_inputs(*parameters.keys())
        
    def generate_schema(self, desc:str, type:str, prefix:str):
            return {'desc': desc, '__dspy_field_type': type, 'prefix': prefix+":"}

    def _set_critique_signature(self, generate_signature):
        prompt = """You're a transparent, honest, intelligent, and unapologetic critic. Your job is to analyze, critique, and then provide feedback about the output of a Large Language Model (LLM).

    Think step-by-step: 
    1- Explain your understanding of the LLMs instructions as provided in `instruction_prompt` field.
        - Explain your understanding of ALL instructions for EACH `target_output_fields`.
    2- Evaluate and critique the LLM's response for EVERY `target_output_fields`.
        - Determine how well EACH of the `target_output_fields`s meet the quality, demands, and standards defined for them in the instructions.
        - For each `target_output_fields`, provide the weaknesses, areas of improvement, and deficiencies with respect to the instructions (`instruction_prompt`).
    3- Based on the critique of step 2., provide feedback to improve the quality for EVERY `target_output_fields` and guide the LLM to generate better outputs that align better with the instructions.
        - Avoid directly providing corrections, as this is the LLM's job. Your job is to provide the critique and feedback that guide the LLM to correct its response.
    """
        
        

        fields = deepcopy(generate_signature.fields)

        # Get output fields
        output_fields = []
        for field_name, field in fields.items():
            if field.json_schema_extra['__dspy_field_type'] == 'output':
                output_fields.append(field_name)
                fields[field_name].json_schema_extra['__dspy_field_type'] = 'input'

            fields[field_name] = (field.annotation, field)

        
        # Create the instruction prompt field with proper default
        instruction_prompt_field= pydantic.Field(generate_signature.__doc__, json_schema_extra=self.generate_schema(desc="The instructions given to the Large Language Model as a prompt.", type='input', prefix="Instruction Prompt"))
        target_output_fields_field= pydantic.Field(output_fields.copy(), json_schema_extra=self.generate_schema(desc="Fields to provide critique and feedback about.", type='input', prefix="Target Output Fields"))

        critique= pydantic.Field( json_schema_extra=self.generate_schema(desc="Detailed critique of the output.", type='output', prefix="Critique"))
        feedback= pydantic.Field( json_schema_extra=self.generate_schema(desc="Detailed feedback based on the critique without directly providing corrections.", type='output', prefix="Feedback"))

        fields.update({
            '__doc__': prompt,
            'instruction_prompt': (str,instruction_prompt_field),
            'target_output_fields': (List[str], target_output_fields_field),
            'critique': (str, critique),
            'feedback': (str, feedback),
        })

        critique_signature = pydantic.create_model(
            "CritiqueSignature",
            __base__=dspy.Signature,
            **fields,
        )

        self.critique = dspy.Predict(critique_signature)


    def _set_refine_signature(self, generate_signature):
        prompt = """You're a professional and intelligent editor and refiner. Your job is to provide improvements and correction to a Large Language Model's output given based on defined critique and feedback.

    Think step-by-step: 
    1- Explain your understanding of the instructions that were given to the LLM as provided in `instruction_prompt` field.
        - Explain your understanding of ALL instructions for EACH `target_output_fields`.
    2- Explain your understanding of the `critique` and `feedback` to the LLM's response for EVERY `target_output_fields` and how they relate to the `instruction_prompt`.
    3- Based on the `critique` and `feedback` of step 2., make corrections and improvements to the original `target_output_fields`."""

        fields = deepcopy(generate_signature.fields)

        # Get output fields
        output_fields = []
        for field_name, field in fields.items():
            if field.json_schema_extra['__dspy_field_type'] == 'output':
                output_fields.append(field_name)
                fields[field_name].json_schema_extra['__dspy_field_type'] = 'input'

            fields[field_name] = (field.annotation, field)

        
        # Create the instruction prompt field with proper default
        instruction_prompt_field= pydantic.Field(generate_signature.__doc__, json_schema_extra=self.generate_schema(desc="The instructions given to the Large Language Model as a prompt.", type='input', prefix="Instruction Prompt"))
        
        target_output_fields_field= pydantic.Field(output_fields.copy(), json_schema_extra=self.generate_schema(desc="Fields to provide critique and feedback about.", type='input', prefix="Target Output Fields"))
        
        critique= pydantic.Field( json_schema_extra=self.generate_schema(desc="critique of the output.", type='input', prefix="Critique"))
        feedback= pydantic.Field( json_schema_extra=self.generate_schema(desc="Feedback to improve the output.", type='input', prefix="Feedback"))

        fields.update({
            '__doc__': prompt,
            'instruction_prompt': (str,instruction_prompt_field),
            'target_output_fields': (List[str], target_output_fields_field),
            'critique': (str, critique),
            'feedback': (str, feedback),
        })
        

        refining_fields = {
            'refined_' + field: (generate_signature.model_fields.get(field).annotation, deepcopy(generate_signature.model_fields.get(field)))
            for field in output_fields
        }

        for field_name, field_value in refining_fields.items():
            refining_fields[field_name][1].json_schema_extra['desc'] = field_value[1].json_schema_extra['desc'].replace('${', '${refined_')
            refining_fields[field_name][1].json_schema_extra['prefix'] = 'Refined ' + field_value[1].json_schema_extra['prefix']
        
        fields.update(refining_fields)

        refine_signature = pydantic.create_model(
            "RefineSignature",
            __base__=dspy.Signature,
            **fields,
        )

        self.refine = dspy.Predict(refine_signature)