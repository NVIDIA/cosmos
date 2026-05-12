# Prompting

> **Skill:** `.agents/skills/cosmos3-inference/SKILL.md`

<!--TOC-->

______________________________________________________________________

**Table of Contents**

- [Vision Generation](#vision-generation)
  - [Do](#do)
  - [Don't](#dont)
  - [Upsampling](#upsampling)
- [Action Generation](#action-generation)
  - [Do](#do-1)

______________________________________________________________________

<!--TOC-->

## Vision Generation

**Modalities:** Text2Video, Image2Video, Video2Video

Write your prompt as a **rich, flowing narrative paragraph** that describes the scene exactly as it unfolds.

Example:

> The video begins with a view from inside a vehicle, likely captured by a dashboard camera, showing a wide street scene under an overcast sky. The road stretches ahead with multiple lanes, and several vehicles are visible, including a white car directly ahead and a larger bus to the right. The bus has an advertisement for a law firm on its side. On the left side of the street, there's a parking lot filled with cars, and various commercial buildings line either side, featuring signs and storefronts. The environment suggests a suburban or semi-urban area with palm trees scattered along the sidewalks, adding a touch of greenery to the otherwise urban landscape. As the video progresses, the vehicle continues to move forward down the street. The white car directly ahead remains in the same lane, maintaining a steady pace. The bus on the right side of the road begins to look closer to the ego vehicle, as the large size becomes more prominent while moving towards it. The advertisements, including one for a law firm, become clearer as the ego vehicle overtakes a bus stopped ahead. The surrounding environment remains consistent, with the parking lot on the left and commercial buildings on the right under the same cloudy sky. By the final frame, both the white car and the ego vehicle move forward from the bus, making it out of the frame, and the white car slightly turns right, with its right blinker on, continuing to show the same street with the same vehicles and surroundings, maintaining the calm and steady pace of the journey.

[More examples](../inputs/t2v_long_prompts.jsonl)

### Do

1. **Describe actions chronologically**, grounded in spatial landmarks
   - *"A woman walks through a wooden doorway"* rather than *"a woman comes into view"*
1. **Be highly specific** about visible details:
   - Subject appearance, clothing, and movements
   - Lighting and cinematic camera angles
   - Overall atmospheric aesthetic
1. **Use the subject's perspective** for left/right directions
1. **Stick to objective, visible elements** — describe only what the camera can see
1. **Be vivid and precise** about the physical scene, environment, and temporal actions

### Don't

1. Use meta-phrases like *"generate a video of..."* or *"enters the frame"*
1. Reference invisible backstories or hidden emotions
1. Write abstract or vague descriptions

> **Key takeaway:** The more vividly and precisely you describe the physical scene, environment, and temporal actions, the better the model can bring your vision to life.

### Upsampling

Prompt upsampling is strongly recommended. We provide an [upsampling template](../cosmos3/defaults/prompt_upsampler.txt) that is known to work well with [Qwen/Qwen3-VL-8B-Instruct-FP8](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-FP8).

To run locally, start a vLLM server. This will take a few minutes.

```shell
uvx --with nvidia-cuda-runtime-cu12 \
vllm@0.19.0 serve Qwen/Qwen3-VL-8B-Instruct-FP8 \
--tensor-parallel-size 1
```

The server is ready when you see `Application startup complete.`

In a separate terminal, run the following. The outputs will be in the specified path.

```shell
python -m cosmos3.scripts.upsample_prompts -i "inputs/omni/*.json" -o outputs/upsample_prompts
```

## Action Generation

**Modalities:** Forward Dynamics, Inverse Dynamics, Policy

Action generation prompts should be concise. Example:

> Put the pot to the left of the purple item. This video is captured from a first-person perspective looking at the scene.

### Do

1. Use precise spatial language
Specify exact directional relationships ("to the left of") rather than vague terms like "near" or "next to." Relative positioning anchors the action clearly in 3D space.
1. Reference objects by distinct attributes
Identify objects by a unique, visible property ("the purple item") rather than just category names. Color, size, and shape help the model disambiguate when multiple similar objects are present.
1. State the action as a direct imperative
Begin with a verb in command form ("Put," "Move," "Place"). This keeps the prompt unambiguous about what should happen versus what is being described.
1. Specify the camera perspective explicitly
Include a perspective clause ("captured from a first-person perspective looking at the scene"). The same action looks very different from a top-down, third-person, or egocentric view, and models benefit from this being stated rather than inferred.
1. Keep the sentence structure simple and linear
One action, one object, one target location. Compound instructions ("pick up X, then put it left of Y, while facing Z") introduce ambiguity and increase failure modes.
1. Ground spatial references to the scene, not the viewer
"Left of the purple item" is scene-anchored. "To your left" or "in front of you" can shift meaning depending on assumed viewer orientation. Scene-anchored references are more stable.
1. Avoid hedging or descriptive filler
Phrases like "try to place" or "it would be nice if" weaken the instruction signal. Be declarative and specific.
