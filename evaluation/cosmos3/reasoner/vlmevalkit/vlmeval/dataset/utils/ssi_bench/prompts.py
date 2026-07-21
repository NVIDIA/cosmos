from __future__ import annotations

GROUND_HEIGHT_PROMPT_TEMPLATE = """\
You are a capable vision-based reasoning agent designed to analyze structural
components and infer spatial relationships from images. All geometric judgments
must be inferred in the **three-dimensional structure** (world coordinates)
rather than measured from the 2D image projection. Your goal is to **rank
structural members based on the heights of their centroids relative to the
ground plane**.

**Your Task**

You will be provided with an **Original Structure Image** and a set of shuffled
**Annotated Member Images** (labeled 1, 2, 3, 4). In each annotated image, the
visible portion of the corresponding member is highlighted as a **{color}
region**. The highlighted region may be incomplete due to occlusion or because
part of the member lies outside the image frame.

When estimating the centroid, treat each member as a **complete structural
unit**. If only part of a member is visible, infer the centroid based on the
**smallest complete member unit that the visible region can reasonably
represent**.

Sort the members according to their centroid heights in **ascending order**,
from **lowest to highest**.

If two or more members have equal or indistinguishably similar centroid heights,
the member with the **smaller numerical label** should appear first.

**Output Format**

Your response **must be only** a Python list of integers representing the order
of member labels from lowest to highest. Do not include any other text,
reasoning, or explanation.

**Example**: If you determine the order is Member 3 (lowest), Member 4,
Member 1, Member 2 (highest), your output must be:
[3, 4, 1, 2]

Now please provide your answer in the requested format.
"""


GROUND_ANGLE_PROMPT_TEMPLATE = """\
You are a capable vision-based reasoning agent designed to analyze structural
components and infer geometric relationships from images. All geometric
judgments must be inferred in the **three-dimensional structure** (world
coordinates) rather than measured from the 2D image projection. Your goal is to
**rank structural members according to the angles between their main directions
and the ground plane**.

**Your Task**

You will be provided with an **Original Structure Image** and a set of shuffled
**Annotated Member Images** (labeled 1, 2, 3, 4). In each annotated image, the
visible portion of the corresponding member is highlighted as a **{color}
region**. The highlighted region may be incomplete due to occlusion or because
part of the member lies outside the image frame.

When estimating the angle, treat each member as a **complete structural unit**.
If only part of a member is visible, infer the main direction based on the
**smallest complete member unit that the visible region can reasonably
represent**.

Sort the members according to the angle between their main direction and the
ground plane in **ascending order**, from **smallest angle to largest angle**.
A member **parallel to the ground plane** has the **smallest angle**, while a
member **perpendicular to the ground plane** has the **largest angle**.

If two or more members have equal or indistinguishably similar angles, the
member with the **smaller numerical label** should appear first.

**Output Format**

Your response **must be only** a Python list of integers representing the order
of member labels from smallest angle to largest angle. Do not include any other
text, reasoning, or explanation.

**Example**: If you determine the order is Member 2 (smallest angle),
Member 1, Member 4, Member 3 (largest angle), your output must be:
[2, 1, 4, 3]

Now please provide your answer in the requested format.
"""


DIMENSION_PROMPT_TEMPLATE = """\
You are a capable vision-based reasoning agent designed to analyze structural
components and infer geometric properties from images. All geometric judgments
must be inferred in the **three-dimensional structure** (world coordinates)
rather than measured from the 2D image projection. Your goal is to **rank
structural members according to their lengths along their main directions**.

**Your Task**

You will be provided with an **Original Structure Image** and a set of shuffled
**Annotated Member Images** (labeled 1, 2, 3, 4). In each annotated image, the
visible portion of the corresponding member is highlighted as a **{color}
region**. The highlighted region may be incomplete due to occlusion or because
part of the member lies outside the image frame.

For each member, consider its **dimension as the length measured along its main
(dominant) direction**, not its width, thickness, or projected size in other
directions.

When estimating the dimension, treat each member as a **complete structural
unit**. If only part of a member is visible or occluded, infer the length based
on the **smallest complete member unit that the visible region can reasonably
represent**.

Sort the members according to their dimensions in **ascending order**, from
**shortest to longest**.

If two or more members have equal or indistinguishably similar dimensions, the
member with the **smaller numerical label** should appear first.

**Output Format**

Your response **must be only** a Python list of integers representing the order
of member labels from shortest to longest. Do not include any other text,
reasoning, or explanation.

**Example**: If you determine the order is Member 4 (shortest), Member 1,
Member 3, Member 2 (longest), your output must be:
[4, 1, 3, 2]

Now please provide your answer in the requested format.
"""


RELATIVE_DISTANCE_PROMPT_TEMPLATE = """\
You are a capable vision-based reasoning agent designed to analyze structural
components and infer spatial relationships from images. All geometric judgments
must be inferred in the **three-dimensional structure** (world coordinates)
rather than measured from the 2D image projection. Your goal is to **rank each
group based on the relative distance between the two structural members it
contains**.

**Your Task**

You will be provided with an **Original Structure Image** and a set of shuffled
**Annotated Group Images** (labeled 1, 2, 3). Each annotated group contains
**two structural members**, with the visible portions of both members
highlighted as **{color} regions**. The highlighted regions may be incomplete
due to occlusion or because parts of the members lie outside the image frame.

For each group, consider the **relative distance between the two members**,
defined as the **shortest distance between the infinite straight lines that
coincide with the main (dominant) directions of the two members**.

If the two lines **intersect**, their relative distance is defined as **0**.

Sort the groups according to their relative distances in **ascending order**,
from **smallest distance to largest distance**.

If two or more groups have equal or indistinguishably similar distances, the
group with the **smaller numerical label** should appear first.

**Output Format**

Your response **must be only** a Python list of integers representing the order
of group labels from smallest distance to largest distance. Do not include any
other text, reasoning, or explanation.

**Example**: If you determine the order is Group 3 (smallest distance),
Group 1, Group 2 (largest distance), your output must be:
[3, 1, 2]

Now please provide your answer in the requested format.
"""

AREA_PROMPT_TEMPLATE = """\
You are a capable vision-based reasoning agent designed to analyze structural
components and infer geometric properties from images. All geometric judgments
must be inferred in the **three-dimensional structure** (world coordinates)
rather than measured from the 2D image projection. Your goal is to **rank
groups according to the areas of planar convex polygons formed by their
nodes**.

**Your Task**

You will be provided with an **Original Structure Image** and a set of shuffled
**Annotated Group Images** (labeled 1, 2, 3). Each annotated group contains a
**set of nodes**, highlighted as **{color} points**, which together define a
planar convex polygon.

For each group, consider the **area of the planar convex polygon formed by the
given set of nodes**, i.e., the **area of the convex hull of the nodes**.

Sort the groups according to their polygon areas in **ascending order**, from
**smallest area to largest area**.

If two or more groups have equal or indistinguishably similar areas, the group
with the **smaller numerical label** should appear first.

**Output Format**

Your response **must be only** a Python list of integers representing the order
of group labels from smallest area to largest area. Do not include any other
text, reasoning, or explanation.

**Example**: If you determine the order is Group 2 (smallest area), Group 1,
Group 3 (largest area), your output must be:
[2, 1, 3]

Now please provide your answer in the requested format.
"""


VOLUME_PROMPT_TEMPLATE = """\
You are a capable vision-based reasoning agent designed to analyze structural
components and infer geometric properties from images. All geometric judgments
must be inferred in the **three-dimensional structure** (world coordinates)
rather than measured from the 2D image projection. Your goal is to **rank
groups according to the volumes of convex polyhedra formed by their nodes**.

**Your Task**

You will be provided with an **Original Structure Image** and a set of shuffled
**Annotated Group Images** (labeled 1, 2, 3). Each annotated group contains a
**set of nodes**, highlighted as **{color} points**, which together define a
three-dimensional shape.

For each group, consider the **volume of the convex polyhedron formed by the
given set of nodes**, i.e., the **volume of the 3D convex hull of the nodes**.

Sort the groups according to their polyhedron volumes in **ascending order**,
from **smallest volume to largest volume**.

If two or more groups have equal or indistinguishably similar volumes, the
group with the **smaller numerical label** should appear first.

**Output Format**

Your response **must be only** a Python list of integers representing the order
of group labels from smallest volume to largest volume. Do not include any
other text, reasoning, or explanation.

**Example**: If you determine the order is Group 1 (smallest volume), Group 3,
Group 2 (largest volume), your output must be:
[1, 3, 2]

Now please provide your answer in the requested format.
"""


HOP_DISTANCE_PROMPT_TEMPLATE = """\
You are a capable vision-based reasoning agent designed to analyze structural
connectivity and infer **topological relationships** between structural
members. All topological judgments must be inferred in the **three-dimensional
structure** (world coordinates) rather than measured from the 2D image
projection. Your goal is to **rank each group based on the topological
distance (number of hops) between the two structural members it contains**.

**Your Task**

You will be provided with an **Original Structure Image** and a set of shuffled
**Annotated Group Images** (labeled 1, 2, 3). Each annotated group contains
**two structural members**, with the visible portions of both members
highlighted as **{color} regions**. The highlighted regions may be incomplete
due to occlusion or because parts of the members lie outside the image frame.

When analyzing structural members, treat each member as a **complete structural
unit**. If a structural member is only partially visible, infer it as a
**complete structural unit**, using the **smallest complete member unit**
consistent with the visible region.

Split continuous elements at every connection node; each segment is a member.
Do not merge multi-node segments unless explicitly merged in the annotation.

For each group, determine the **topological distance (hops)** between the two
members, defined as:

- A **hop** is one direct connection between two structural members (e.g.,
  physical joint, intersection, or direct attachment).
- If the two members are **directly connected**, their topological distance is
  **1 hop**.
- If the two members are **not directly connected**, the topological distance
  is the **minimum number of intermediate members** required to form a
  continuous connection path between them.

Sort the groups according to their topological distances in **ascending
order**, from **smallest number of hops to largest number of hops**.

If two or more groups have equal or indistinguishably similar topological
distances, the group with the **smaller numerical label** should appear first.

**Output Format**

Your response **must be only** a Python list of integers representing the order
of group labels from smallest topological distance to largest. Do not include
any other text, reasoning, or explanation.

**Example**: If you determine the order is Group 2 (0 hops), Group 1 (1 hop),
Group 3 (2 hops), your output must be:
[2, 1, 3]

Now please provide your answer in the requested format.
"""


CYCLE_LENGTH_PROMPT_TEMPLATE = """\
You are a capable vision-based reasoning agent designed to analyze structural
connectivity and infer **cyclic topological relationships** among structural
members. All topological judgments must be inferred in the **three-dimensional
structure** (world coordinates) rather than measured from the 2D image
projection. Your goal is to **rank each group based on the minimum number of
edges in a cycle that includes all given members**.

**Your Task**

You will be provided with an **Original Structure Image** and a set of shuffled
**Annotated Group Images** (labeled 1, 2, 3). Each annotated group contains a
**set of structural members**, with the visible portions of all members
highlighted as **{color} regions**. The highlighted regions may be incomplete
due to occlusion or because parts of the members lie outside the image frame.

When analyzing structural members, treat each member as a **complete structural
unit**. If a structural member is only partially visible, infer it as a
**complete structural unit**, using the **smallest complete member unit**
consistent with the visible region.

Split continuous elements at every connection node; each segment is a member.
Do not merge multi-node segments unless explicitly merged in the annotation.

For each group, determine the **minimum cycle length**, defined as:

- A **cycle** is a closed topological path formed by connected structural
  members, where the start and end member coincide.
- The **cycle length** is the total number of distinct edges (connections) in
  the cycle.
- The cycle must **include all members in the group** (each member must lie on
  the cycle).
- If multiple such cycles exist, use the one with the **smallest number of
  edges**.
- If no cycle exists that includes all given members, treat the cycle length as
  **infinite**.

Sort the groups according to their minimum cycle lengths in **ascending
order**, from **smallest number of edges to largest number of edges**.

If two or more groups have equal or indistinguishably similar cycle lengths,
the group with the **smaller numerical label** should appear first.

**Output Format**

Your response **must be only** a Python list of integers representing the order
of group labels from smallest cycle length to largest. Do not include any other
text, reasoning, or explanation.

**Example**: If you determine the order is Group 1 (3 edges), Group 3 (4
edges), Group 2 (no valid cycle), your output must be:
[1, 3, 2]

Now please provide your answer in the requested format.
"""


MV_RELATIVE_DISTANCE_PROMPT_TEMPLATE = """\
You are a capable vision-based reasoning agent designed to analyze structural
components and infer spatial relationships from images. All geometric judgments
must be inferred in the **three-dimensional structure** (world coordinates)
rather than measured from the 2D image projection. Your goal is to **rank
multiple structural members based on their relative distance to a reference
member**, using information from **multiple viewpoints**.

**Your Task**

You will be provided with images from **two different viewpoints** of the same
structure:

- **Two Original Structure Images**, each captured from a different view.
- A set of **Annotated Member Images**:
  - One annotated image highlights **Member 0**, which serves as the
    **reference member**.
  - Three other annotated images highlight **Member 1**, **Member 2**, and
    **Member 3**, respectively.

In each annotated image, the visible portion of the highlighted member is
marked as a **{color} region**. The highlighted regions may be incomplete due
to occlusion or because parts of the members lie outside the image frame.

**Distance Definition**

For each of **Member 1, Member 2, and Member 3**, consider its **relative
distance to Member 0**, defined as the **shortest distance between the infinite
straight line coinciding with the main (dominant) direction of Member 0 and the
infinite straight line coinciding with the main (dominant) direction of the
other member**.

If the two lines **intersect**, their relative distance is defined as **0**.

All distance judgments must integrate information from **both viewpoints** to
reason about the true 3D spatial relationships.

**Sorting Requirement**

Sort **Member 1, Member 2, and Member 3** according to their relative
distances to **Member 0**, in **ascending order**, from **smallest distance to
largest distance**.

If two or more members have equal or indistinguishably similar distances, the
member with the **smaller numerical label** should appear first.

**Output Format**

Your response **must be only** a Python list of integers representing the order
of member labels from smallest distance to largest distance. Do not include any
other text, reasoning, or explanation.

**Example**: If you determine that Member 2 is closest to Member 0, followed by
Member 1, and then Member 3 (farthest), your output must be:
[2, 1, 3]

Now please provide your answer in the requested format.
"""


MV_HOP_DISTANCE_PROMPT_TEMPLATE = """\
You are a capable vision-based reasoning agent designed to analyze structural
connectivity and infer **topological relationships** between structural
members. All topological judgments must be inferred in the **three-dimensional
structure** (world coordinates) rather than measured from the 2D image
projection. Your goal is to **rank multiple structural members based on their
topological distance (number of hops) to a reference member**, using
information from **multiple viewpoints**.

**Your Task**

You will be provided with images from **two different viewpoints** of the same
structure:

- **Two Original Structure Images**, each captured from a different view.
- A set of **Annotated Member Images**:
  - One annotated image highlights **Member 0**, which serves as the
    **reference member**.
  - Three other annotated images highlight **Member 1**, **Member 2**, and
    **Member 3**, respectively.

In each annotated image, the visible portion of the highlighted member is
marked as a **{color} region**. The highlighted regions may be incomplete due
to occlusion or because parts of the members lie outside the image frame.

When analyzing structural members, treat each member as a **complete structural
unit**. If a structural member is only partially visible, infer it as a
**complete structural unit**, using the **smallest complete member unit**
consistent with the visible region.

Split continuous elements at every connection node; each segment is a member.
Do not merge multi-node segments unless explicitly merged in the annotation.

**Topological Distance Definition**

For each of **Member 1, Member 2, and Member 3**, determine its **topological
distance (number of hops)** to **Member 0**, defined as follows:

- A **hop** is one direct connection between two structural members (e.g.,
  physical joint, intersection, or direct attachment).
- If a member is **directly connected** to Member 0, the topological distance
  is **1 hop**.
- If a member is **not directly connected** to Member 0, the topological
  distance is the **minimum number of intermediate members** required to form a
  continuous connection path between the member and Member 0.

All topological judgments must integrate information from **both viewpoints**
to reason about the true 3D structural connectivity.

**Sorting Requirement**

Sort **Member 1, Member 2, and Member 3** according to their topological
distances to **Member 0**, in **ascending order**, from **smallest number of
hops to largest number of hops**.

If two or more members have equal or indistinguishably similar topological
distances, the member with the **smaller numerical label** should appear first.

**Output Format**

Your response **must be only** a Python list of integers representing the order
of member labels from smallest topological distance to largest. Do not include
any other text, reasoning, or explanation.

**Example**: If you determine that Member 1 is directly connected to Member 0
(1 hop), Member 3 is connected via one intermediate member (2 hops), and Member
2 is farther away (3 hops), your output must be:
[1, 3, 2]

Now please provide your answer in the requested format.
"""


MV_CYCLE_LENGTH_PROMPT_TEMPLATE = """\
You are a capable vision-based reasoning agent designed to analyze structural
connectivity and infer **cyclic topological relationships** among structural
members. All topological judgments must be inferred in the **three-dimensional
structure** (world coordinates) rather than measured from the 2D image
projection. Your goal is to **rank multiple structural members based on the
minimum cycle length of a cycle that includes the reference member and the
target member**, using information from **multiple viewpoints**.

**Your Task**

You will be provided with images from **two different viewpoints** of the same
structure:

- **Two Original Structure Images**, each captured from a different view.
- A set of **Annotated Member Images**:
  - One annotated image highlights **Member 0**, which serves as the
    **reference member**.
  - Three other annotated images highlight **Member 1**, **Member 2**, and
    **Member 3**, respectively.

In each annotated image, the visible portion of the highlighted member is
marked as a **{color} region**. The highlighted regions may be incomplete due
to occlusion or because parts of the members lie outside the image frame.

When analyzing structural members, treat each member as a **complete structural
unit**. If a structural member is only partially visible, infer it as a
**complete structural unit**, using the **smallest complete member unit**
consistent with the visible region.

Split continuous elements at every connection node; each segment is a member.
Do not merge multi-node segments unless explicitly merged in the annotation.

All topological reasoning must integrate information from **both viewpoints**
to infer the true 3D structural connectivity.

**Cycle Length Definition**

For each of **Member 1, Member 2, and Member 3**, determine its **minimum cycle
length with Member 0**, defined as:

- A **cycle** is a closed topological path formed by connected structural
  members, where the start and end member coincide.
- The **cycle length** is the total number of distinct edges (connections) in
  the cycle.
- The cycle must **include both Member 0 and the target member** (the target is
  Member 1 / Member 2 / Member 3).
- If multiple such cycles exist, use the one with the **smallest number of
  edges**.
- If no cycle exists that includes both Member 0 and the target member, treat
  the cycle length as **infinite**.

**Sorting Requirement**

Sort **Member 1, Member 2, and Member 3** according to their minimum cycle
lengths with **Member 0**, in **ascending order**, from **smallest number of
edges to largest number of edges**.

If two or more members have equal or indistinguishably similar cycle lengths,
the member with the **smaller numerical label** should appear first.

**Output Format**

Your response **must be only** a Python list of integers representing the order
of member labels from smallest cycle length to largest. Do not include any
other text, reasoning, or explanation.

**Example**: If you determine that the smallest cycle including Member 0 and
Member 2 has 3 edges, Member 0 and Member 1 has 4 edges, and Member 0 and
Member 3 has no valid cycle, your output must be:
[2, 1, 3]

Now please provide your answer in the requested format.
"""


def prompts_by_task_slug(*, multi_view: bool = False) -> dict[str, str]:
    prompts = {
        "ground_height": GROUND_HEIGHT_PROMPT_TEMPLATE,
        "ground_angle": GROUND_ANGLE_PROMPT_TEMPLATE,
        "dimension": DIMENSION_PROMPT_TEMPLATE,
        "relative_distance": RELATIVE_DISTANCE_PROMPT_TEMPLATE,
        "area": AREA_PROMPT_TEMPLATE,
        "volume": VOLUME_PROMPT_TEMPLATE,
        "hop_distance": HOP_DISTANCE_PROMPT_TEMPLATE,
        "cycle_length": CYCLE_LENGTH_PROMPT_TEMPLATE,
    }

    if multi_view:
        prompts.update(
            {
                "relative_distance": MV_RELATIVE_DISTANCE_PROMPT_TEMPLATE,
                "hop_distance": MV_HOP_DISTANCE_PROMPT_TEMPLATE,
                "cycle_length": MV_CYCLE_LENGTH_PROMPT_TEMPLATE,
            }
        )

    return prompts
