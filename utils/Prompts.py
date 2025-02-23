AGENTS_INSTRUCTION = (
    "Here are the available agents and their functionalities:\n"
    "- ItemRetrievalAgent: Input a recommendation request containing product attributes "
    "(for example:'Please recommend me a sugar-free energy drink.' "
    "It is **invalid** to enter a requirement containing relationship with other specific products here"
    "(such as for a specific product, match with a specific product). However, the type of item to be recommended must be clearly stated. "
    "Such requirement needs to be converted into specific item attributes through SearcherAgent), "
    "and recommend a list containing 10 specific items based on keyword similarity. "
    "If it is for clothing recommendations, men's/women's/boys'/girls' must be retained"
    "A search by SearcherAgent can only retrieve one target for one requirement. "
    "When multiple targets need to be recommended, ItemRetrievalAgent needs to be called multiple times.\n"  # similarity to the text of 
    # "- SearcherAgent: Input keywords no longer than 30 words and search for information you don't know based on user query.\n"
    # "- SearcherAgent: Input a short description of specific usage requirements and search product attributes that meet the needs "
    "- SearcherAgent: Input a short query and search for product attributes that meet the needs "
    "from a knowledge base of product attributes and usage, based on keyword similarity. "
    # "Input can only be usage requirements such as 'what types of shorts are suitable for running', not arbitrary queries."
    "The query target can only be the attributes that meet the target requirements, such as "
    "'what kind of shorts are suitable for mountain climbing'. Note that it is the attributes and not other things. "
    "You cannot find other information through this agent."
    # "This knowledge base only contains the correspondence between common attributes and common uses. "
    # "This knowledge base only contains item attributes and uses. "
    # "Other information cannot be queried.\n"
    "The attributes returned by SearcherAgent are guaranteed to be retrieved in ItemRetrievalAgent. "
    "Even if you think they are not appropriate, they are the closest answers to the search input."
    # "- ChatAgent: Just engage in a general conversation with the user.\n"
    # "- InteractorAgent: Input recommend target and generate a final response with one or more recommend result lists "
    "- InteractorAgent: Input 'Recommend' and generate a final response with one or more recommend result lists "
    "(the input does not need to include the recommended items)"
)
BoT_AGENTS_INSTRUCTION = (
    f"{AGENTS_INSTRUCTION}"
    "- PlannerAgent: Input the re-plan goal(for example, The task history provides a list of available product types. "
    "Select the two most suitable ones and then enter: 'Generate a recommendation plan for type A and type B'. "
    "the types of products must be specific product name, with one recommendation for each type. "
    "The number of product types should not exceed 2. And it should not include products type that have been recommended before.) "# 不超过2
    "and Regenerate subsequent tasks in the same way as the "  
    "initial plan based on the information obtained from the executed subtasks. "
    "It marks the end of a phased plan. This Agent can **ONLY be placed at the end of a phased plan!!!**"
)
CLOTH_RETRIEVE_PROMPT = """The user's query is:{user_input}.
    From this, please extract the user's requirements and preferences for clothing."
    Please fill in this format and only output the filled content:[clothing type]; [preference]. 
    clothing type is the basic attributes and gender distinction. The user's personalized preferences are in preference.
    You can also select a small number of attributes from the user's query and add them to the preference.
    The two parts must be separated by ';'.
    The total length must not exceed 15 words. Separate multiple attributes with ' '.
    The basic attributes of the product are prioritized, followed by the detailed attributes. 
    You only need to reflect the preferences in the user input and preference without making any inferences. """

BEAUTY_RETRIEVE_PROMPT = """The user's query is:{user_input}. 
    From this, please extract the user's requirements and preferences for beauty product."
    Please fill in this format and only output the filled content:[product type]; [preference]. 
    product type is the basic product type and specification(such as capacity or size). The user's personalized preferences are in preference.
    You can also select a small number of attributes from the user's query and add them to the preference.
    The two parts must be separated by ';'.
    The total length must not exceed 10 words. Separate multiple attributes with ' '.
    The basic attributes of the product are prioritized, followed by the detailed attributes. 
    You only need to reflect the preferences in the user input and preference without making any inferences. """

MUSIC_RETRIEVE_PROMPT = """The user's query is:{user_input}. 
    From this, please extract the user's requirements and preferences for music."
    Please fill in this format and only output the filled content:[music type]; [preference]. 
    music type is the basic product type and specification(such as Country or rock). The user's personalized preferences are in preference.
    You can also select a small number of attributes from the user's query and add them to the preference.
    The two parts must be separated by ';'.
    The total length must not exceed 5 words. Separate multiple attributes with ' '.
    The basic attributes of the product are prioritized, followed by the detailed attributes. 
    You only need to reflect the preferences in the user input and preference without making any inferences. """

PRODUCT_RETRIEVE_PROMPT = """The user's query is:{user_input}. From this, please extract the user's requirements and preferences for product."
    Please fill in this format and only output the filled content:[product type]; [preference]. 
    product type is the basic attributes. Other attributes are in preference.
    If there is only information about the product category and no more preference information, the preference part will be blank.
    These are used for text similarity matching and should not contain text that does not appear in the product information.
    The two parts must be separated by ';'.
    The total length must not exceed 15 words. Separate multiple attributes with ' '.
    Include as much of the key points of user requirements as possible, 
    and the basic attributes of the product are prioritized, followed by the detailed attributes. 
    You only need to reflect the preferences in the user input without making any inferences. """

ZERO_SHOT_PROMPT = (
    "Based on the user's input and current information, especially the newest agent output, "
    "you need to think about whether you have enough information to make a recommendation."
    "Or whether the retrieval results are suitable for recommendation"
    "And create a next task plan in JSON format with sub-tasks."
    "\nONLY output JSON format as follows, NOTHING ELSE!!!:\n"
    "{ \"agent\": \"...\", \"input\": \"...\" }"
    "'agent' specify the agent to execute next sub-task. 'input' is according to agent functionalities."
    f"{AGENTS_INSTRUCTION}"
)
FEW_SHOT_PROMPT = (
    "Based on the user's input and current information, especially the newest agent output, "
    "you need to think about whether you have enough information to make a recommendation."
    "Or whether the retrieval results are suitable for recommendation"
    "And create a next task plan in JSON format with sub-tasks."
    "\nONLY output JSON format that can be correctly recognised as follows, NOTHING ELSE!!!:\n"
    "{ \"agent\": \"...\", \"input\": \"...\" }"
    "Please remember to output only these, otherwise they will not be recognised"
    "'agent' specify the agent to execute next sub-task. 'input' is according to agent functionalities."
    f"{AGENTS_INSTRUCTION}"
    "Here are some examples:"
    # "latest task history: Recommend me a pair of shoes."
    # "output: { \"agent\": \"InteractorAgent\", \"input\": \"Ask for more information\" }"
    "latest task history: Recommend me a pair of shoes for ballet."
    "output: { \"agent\": \"ItemRetrievalAgent\", \"input\": \"Recommend a pair of shoes for ballet. \" }"
    "latest task history: Please recommend some clothes suitable for an art exhibition."
    "output: { \"agent\": \"SearcherAgent\", \"input\": \"What kind of clothes should be worn at an art exhibition. \" }"
    "latest task history: search:(at an art exhibition)In spring and summer, you can wear simple and casual T-shirts and shirts, and in autumn and winter, you can wear textured windbreakers."
    "output: { \"agent\": \"ItemRetrievalAgent\", \"input\": \"Recommend simple and casual T-shirts and shirts; Recommend textured windbreakers.\" }"
    "latest task history: (ItemRetrievalAgent outputs a list of recommendations that fit the user's query)."
    "output: { \"agent\": \"InteractorAgent\", \"input\": \"recommend to the user\" }"
    "latest task history: (ItemRetrievalAgent outputs a list of recommendations that don't match the user's query)."
    "output: { \"agent\": \"SearcherAgent\", \"input\": \"(Input query for searching information)\" }"
    "(END OF EXAMPLES)"
)
FEW_SHOT_REACT_EXAMPLE = (
    "Here are some examples:\n"
    ""
    "latest task history: Recommend me a pair of shoes for ballet."
    "Observation: The user is looking for shoes for ballet. It's a This is a common requirement."
    "Thought: We need to retrieve specific shoes that match the user's requirement.. The best agent for this task is the ItemRetrievalAgent, as it can retrieve specific items based on the user's reference. The input should be a query that specifies the need for shoes for ballet"
    "Action output: { \"agent\": \"ItemRetrievalAgent\", \"input\": \"Recommend a pair of shoes for ballet. \" \n}"
    ""
    "latest task history: Please recommend some clothes suitable for an art exhibition."
    "Observation: The user is looking for clothes suitable for an art exhibition. It's not clear what kind of clothing is appropriate for an art exhibition."
    "Thought: To refine the search and provide more suitable recommendations, we should use the SearcherAgent to find information. The input should be clothing type that is appropriate for an art exhibition. This will help us guide the ItemRetrievalAgent more effectively in the next step."
    # "Action output: { \"agent\": \"SearcherAgent\", \"input\": \"What kind of clothes should be worn at an art exhibition. \" \n}"
    "Action output: { \"agent\": \"SearcherAgent\", \"input\": \"What kind of clothes are suitable to wear at art exhibition\" \n}"
    ""
    "latest task history: search:(at an art exhibition)In spring and summer, you can wear simple T-shirts and shirts, and in autumn and winter, you can wear textured windbreakers."
    "Observation: The user wants clothes suitable for an art exhibition. The searcherAgent provided insights into clothing suitable for an art exhibition, including simple T-shirts and shirts for spring and summer and textured windbreakers for autumn and winter"
    "Thought: To make a recommendation that fits the user's need, we need to retrieve specific clothing that match the user's criteria using the ItemRetrievalAgent.The input should be simple T-shirts and shirts and textured windbreakers suggested by searcherAgent. "
    "output: { \"agent\": \"ItemRetrievalAgent\", \"input\": \"Recommend simple and casual T-shirts and shirts; Recommend textured windbreakers.\" \n}"
    ""
    "latest task history: (ItemRetrievalAgent outputs a list of recommendations that fit the user's query)."
    "Observation: The user wants clothes suitable for an art exhibition. The ItemRetrievalAgent retrieved simple T-shirts and shirts and textured windbreakers suggested by searcherAgent."
    "Thought: Recommendations can be made to users, given that a recommendation list that meets the user's requirements is already available. The next agent should be InteractorAgent and the input should be recommend to the user."
    "output: { \"agent\": \"InteractorAgent\", \"input\": \"recommend to the user.\" \n}"
    ""
    "latest task history: (ItemRetrievalAgent outputs a list of recommendations that don't match the user's query)."
    "Observation: The user wants clothes suitable for an art exhibition. The ItemRetrievalAgent retrieved clothes with art designs, but this is the wrong direction to recommend."
    "Thought: The information required for the recommendation is not sufficient and further search is needed. We should use the SearcherAgent to find information. The input should be clothing type that is appropriate for an art exhibition. This will help us guide the ItemRetrievalAgent more effectively in the next step."
    # "Action output: { \"agent\": \"SearcherAgent\", \"input\": \"What kind of clothes should be worn at an art exhibition. \" \n}"
    "Action output: { \"agent\": \"SearcherAgent\", \"input\": \"What kind of clothes are suitable to wear at art exhibition\" \n}"
    # "(END OF EXAMPLES)"
)
REACT_PROMPT = (
    "Based on the user's input and current information, especially the newest agent output, "
    "you need to think about whether you have enough information to make a recommendation."
    "Or whether the retrieval results are suitable for recommendation"
    "And create a next task plan in JSON format with sub-tasks."
    ""
    "You must strictly follow the Observation, Thinking, and Action steps to solve this task and record the process."
    "In the thought, you need to think: "
    "whether the information you currently have is enough to make a recommendation that fits the user's need;"
    "Please **avoid repeatedly executing tasks**. If a task has been executed once and you are not satisfied with the result, "
    "executing it again or making minor changes to execute will **not work**!"  #
    "how you get information through the next task;"
    "what kind of input can guide the agent to complete the task accurately."
    "The input format specified in the agent description below must be followed."
    ""
    "\nOutput JSON format as follows, only action is output in json format:\n"
    "{ \"agent\": \"...\", \"input\": \"...\" }"
    "'agent' specify the agent to execute next sub-task. 'input' is according to agent functionalities."
    ""
    f"{AGENTS_INSTRUCTION}"
    "You can only assign one Agent to complete one task. "
    "If you need the same agent to complete multiple tasks, please only output the first task. "
    f"{FEW_SHOT_REACT_EXAMPLE}"
)
INTEREC_PROMPT = f"""
    {REACT_PROMPT}\n
    The following is an example of a task execution.
    Query: I'm looking for the top-rated resveratrol supplement that doesn't include tea leaves. Additionally, could you recommend a good NMN supplement to pair with it?
    Plan: 1. First Recommendation: Top-Rated Resveratrol Supplement Without Tea Leaves
    SearcherAgent
    
    Input: "resveratrol supplement without tea leaves"
    ItemRetrievalAgent
    
    Input: "top-rated resveratrol supplement without tea leaves"
    2. Second Recommendation: Good NMN Supplement to Pair with Resveratrol
    SearcherAgent
    
    Input: "NMN supplement to pair with resveratrol"
    ItemRetrievalAgent
    
    Input: "good NMN supplement to pair with resveratrol"
    3. Compile Recommendations
    InteractorAgent
    Input: "Recommend"
"""
REFLECTION_HEADER = """You have attempted to complete the above task before and failed. 
The following reflection(s) give a plan to avoid failing to complete the task in the same way you did previously. 
Use them to improve your strategy of correctly completing the given task."""

REFLECTION_PROMPT = (
    f"{REACT_PROMPT}"
)

REFLECT_INSTRUCTION = f"""You will be given a previous reasoning trial in which 
you were given access to the product database and online search tool and need to complete a recommendation task.\n 
The task prompt is: {REFLECTION_PROMPT}\n
You were unsuccessful in complete the recommendation either because you provide a list of recommendations 
that doesn't have enough items user satisfied with, or you used up your set number of reasoning steps. 

In a few sentences, Diagnose a possible reason for failure and devise a new, concise, high level plan 
that aims to mitigate the same failure. Use complete sentences.  \n""" + """Previous trial:
Question: {query}
Trial history: {history}

Reflection:"""

ONE_PLAN_PROMPT = (
)