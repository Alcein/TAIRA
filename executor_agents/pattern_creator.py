import re
import json

from utils.task import get_completion


def update_templates(existing_templates, updates):
    """
    Updates the existing templates dictionary with new information from updates.

    Args:
        existing_templates (dict): The current templates.
        updates (dict): The new or updated templates.

    Returns:
        dict: The updated templates.
    """
    for key, value in updates.items():
        if key in existing_templates:
            existing_templates[key].update(value)
        else:
            existing_templates[key] = value

    return existing_templates


def auto_construct_template(task_route, old_pattern):
    sys_prompt = (
        "You are a thought pattern updater and specialize in refining cognitive processes to "
        "enhance performance.\n"
        "The thought pattern is a high-level idea extracted from the task execution path, which "
        "contains the following three parts:\n"
        "1. Task Description: Used to characterize a task, it is abstract and general, does not "
        "include any specific task information, and only describes what type a task belongs to.\n"
        "2. Solution Description: Used to describe the macro idea of solving a problem, it is "
        "more like the key to solving a problem told by a domain expert, so it mainly comes from the opinions of experts.\n"
        "3. Thought Template: A template for a path to complete a task, used to indicate the "
        "generation of a plan.\n"

    )
    prompt = (
        f"Based on the given successful task route, consider how to adjust the old pattern: \n{old_pattern}.\n"
        f"Task Route: {task_route}\n"
        "The output should strictly align with the structure and tone of the old pattern, maintaining "
        "clarity and effectiveness.\n"
        "Provide the revised thought pattern as a concise, actionable statement that improves "
        "upon the old approach."
    )

    messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt}]
    response = get_completion(messages)  # Simulate LLM call
    return response

def construct_template(task_route, expert_opinion, old_pattern):
    sys_prompt = (
        "You are a thought pattern creator and specialize in refining cognitive processes to "
        "enhance performance.\n"
        "The thought pattern is a high-level idea extracted from the task execution path, which "
        "contains the following three parts:\n"
        "1. Task Description: Used to characterize a task, it is abstract and general, does not "
        "include any specific task information, and only describes what type a task belongs to.\n"
        "2. Solution Description: Used to describe the macro idea of solving a problem, it is "
        "more like the key to solving a problem told by a domain expert, so it mainly comes from the opinions of experts.\n"
        "3. Thought Template: A template for a path to complete a task, used to indicate the "
        "generation of a plan.\n"

    )
    prompt = (
        f"Based on the given successful task route and expert opinion, consider how to create a new pattern. \n"
        f"Task Route: {task_route}\n"
        f"Expert Opinion: {expert_opinion}\n"
        f"The output should strictly align with the structure and tone of the example pattern:{old_pattern}, "
        "maintaining clarity and effectiveness.\n"
        "Provide the revised thought pattern as a concise, actionable statement that improves "
        "upon the old approach."
    )

    messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt}]
    response = get_completion(messages)  # Simulate LLM call
    return response

def correct_template(task_route, expert_correction, old_pattern):
    sys_prompt = (
        "You are a thought pattern creator and specialize in refining cognitive processes to "
        "enhance performance.\n"
        "The thought pattern is a high-level idea extracted from the task execution path, which "
        "contains the following three parts:\n"
        "1. Task Description: Used to characterize a task, it is abstract and general, does not "
        "include any specific task information, and only describes what type a task belongs to.\n"
        "2. Solution Description: Used to describe the macro idea of solving a problem, it is "
        "more like the key to solving a problem told by a domain expert, so it mainly comes from the opinions of experts.\n"
        "3. Thought Template: A template for a path to complete a task, used to indicate the "
        "generation of a plan.\n"

    )
    prompt = (
        f"Based on the given failed task route and expert correction, consider how to adjust the old pattern: \n{old_pattern}.\n"
        f"Task Route: {task_route}\n"
        f"Expert Correction: {expert_correction}\n"
        "The output should strictly align with the structure and tone of the old pattern, maintaining "
        "clarity and effectiveness.\n"
        "Provide the revised thought pattern as a concise, actionable statement that improves "
        "upon the old approach."
    )

    messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt}]
    response = get_completion(messages)  # Simulate LLM call
    return response

# Load existing templates from thought_template.py
existing_templates = {
    "template_1": {
        "task_description": """
        In this conversational recommendation query, user puts forward one usage requirement and specifies one 
        clothing type for recommendation (Men's/Women's/Boys'/Girls' clothing is not a specification of usage requirement).
        """,
        "solution_description": """
        To complete the recommendation, you need to generate an execution plan that can successfully generate the recommendation. 
        In order to complete the recommendation, you need to search for items that meet the target attributes 
        in the item database through keywords. However, for some complex user needs, the item attributes are unclear. 
        At this time, you need to call the searcher to search for relevant knowledge and obtain the item attributes 
        that can meet the user's needs.
        """,
        "thought_template": """
        Step 1: Determine the user's target products and needs.\n
        Step 2: Determine whether there are clear product attributes for item retrieval. If not, you need to obtain relevant knowledge through searcherAgent.\n
        Step 3: Use the obtained item attributes to retrieve items in ItemRetriever.\n
        Step 4: Recommend the retrieved items to the user.\n
        """
    }
}

# New updates to templates
new_updates = {
    "template_11": {
        "task_description": """
        Please explain how to use ChatGPT effectively for learning and productivity.
        """,
        "solution_description": """
        To effectively use ChatGPT, users should clearly define their objectives and provide as much detail as possible 
        when asking questions or describing their needs. Leveraging follow-up questions and asking for clarifications can 
        enhance the value of interactions.
        """,
        "thought_template": """
        Step 1: Understand the user's specific objectives for using ChatGPT.\n
        Step 2: Guide the user on structuring their prompts to achieve the best results.\n
        Step 3: Encourage iterative refinement of questions for better clarity and detail.\n
        Step 4: Provide examples or demonstrations if necessary to showcase how ChatGPT can be utilized.
        """
    }
}

expert_experience = """In this task, the user will specify a particular type of clothing and ask for recommendations of other clothing items that can be paired with it. 
To meet the user's needs, you first need to identify the recommended item based on their request. 
Then, consider the attributes of that item and the user's requirements to figure out what kind of clothing would pair well with it. Once you've got that figured out, suggest the matching clothes accordingly."""

expert_correction = """
The reason why recommender failed was that it ignored the user’s second sentence: I’m not sure about the specific wearing scene. 
Since the user is not clear about the specific wearing scene, in order to satisfy the user in one recommendation, 
it is necessary to give recommendations in various possible directions. First, information should be collected to determine 
which categories (corresponding to different specific scenarios) the product attributes that meet the user's needs can be divided into.
Then, recommendation planning should be carried out for each specific situation, and a corresponding recommendation list should be given.
"""

# Update the templates
updated_templates = update_templates(existing_templates, new_updates)

# Save the updated templates to a JSON file
with open('updated_thought_templates.json', 'w') as file:
    json.dump(updated_templates, file, indent=4)

print("Templates have been successfully updated and saved to 'updated_thought_templates.json'")
