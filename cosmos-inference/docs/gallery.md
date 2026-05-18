# Example Gallery

<!--TOC-->

______________________________________________________________________

**Table of Contents**

- [Text2Video](#text2video)
- [Image2Video](#image2video)
- [Forward Dynamics](#forward-dynamics)

______________________________________________________________________

<!--TOC-->

## Text2Video

<details><summary><b>Input prompt</b></summary>

> The video opens with a view of a well-lit indoor space featuring a wooden display case with compartments filled with various fruits, including bananas, apples, pears, oranges, and carambolas. The bananas are neatly arranged in the middle compartment, while apples are in the left and a mix of pears, oranges, and carambolas are in the right. Two robotic arms with grippers are positioned at the bottom of the frame, with the one on the left remaining stationary, partially obscuring the apples. The robotic arm on the right begins its action, extending towards the right side of the display case. It carefully picks up a pear from the fruit section, placing it into a plastic bag in the shopping cart nearby, which has red handles. After securing the pear, the arm retracts back to its original position. The process repeats as the robotic arm picks up an orange and places it in the bag, followed by a carambola. The final frame captures the robotic arm returning to its initial position, leaving the display case and surrounding area unchanged. The video showcases a seamless and efficient automated fruit-picking process, highlighting the precision and efficiency of modern robotics in a retail setting.

</details>

<details><summary><b>Additional Parameters</b></summary>

- checkpoint-path=Cosmos3-Nano
- num_steps=50
- shift=10.0
- seed=5

</details>

| Output Video                                                                                                                                   |
| ---------------------------------------------------------------------------------------------------------------------------------------------- |
| <video src="https://github.com/user-attachments/assets/b79c70be-f93c-45fb-a43f-ce71008ee7b2" width="600" controls autoplay loop muted></video> |

## Image2Video

<details><summary><b>Input prompt</b></summary>

> The video opens with a view of a testing environment, characterized by a large wooden table at the center. On this table, two robot arms are positioned at opposite ends, with the left arm closer to the camera and the right arm further away. Between the hands lies a dark wooden shelf with a red spherical object on its top rack, likely serving as a platform or obstacle. In the background, various pieces of equipment, including a tripod, a chair, are visible. A person wearing a blue jacket and black pants stands near the center of the room, observing the experiment, with a static hand position throughout. The floor is tiled with a patterned design, and additional items like a small robot figure and some cables can be seen scattered around the space. As the video progresses, the right robotic hand extends outward, moving from its initial position towards the red spherical object on the shelf. The hand then picks up the object and places it on the lowest rack of the shelf, completing a smooth, deliberate manipulation. The left robotic hand remains stationary throughout the sequence. No new objects appear in the video; all existing elements maintain their positions except for the movement of the right robotic hand. The scene concludes with the right robotic hand returning to its initial position, while the left hand continues to rest on the table. The overall environment remains unchanged, with the focus remaining on the interaction between the robotic hands and the wooden block, highlighting precise control during the demonstration.

</details>

<details><summary><b>Additional Parameters</b></summary>

- checkpoint-path=Cosmos3-Super
- num_steps=50
- shift=5.0
- seed=0

</details>

| Input Image                                                                                             | Output Video                                                                                                                                   |
| ------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| <img src="https://github.com/user-attachments/assets/ee40515d-ec43-4334-9563-3cdcb3f0653d" width="500"> | <video src="https://github.com/user-attachments/assets/75e8bedb-d1a3-4621-a5b1-96bbeecef286" width="500" controls autoplay loop muted></video> |

## Forward Dynamics

Action inference uses SO(3) pose representations.

| Conditioning       | Output Video                                                                                                                                   |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| Camera conditioned | <video src="https://github.com/user-attachments/assets/b9b3e46e-4da5-4804-8d1f-f7f8ba31bf90" width="600" controls autoplay loop muted></video> |
| Action conditioned | <video src="https://github.com/user-attachments/assets/be612603-607a-43b0-ac12-dde5bf673ea5" width="600" controls autoplay loop muted></video> |
| Action conditioned | <video src="https://github.com/user-attachments/assets/0629dfe0-c457-4196-8fcb-ca0419ca828c" width="600" controls autoplay loop muted></video> |
