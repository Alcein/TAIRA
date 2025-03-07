thought_templates = {
    "template_0": {
        "task_description": """
        In this conversational recommendation query, user puts forward one item requirement and specifies one 
        type for recommendation(including cloth, beauty and music).
        All the users mentioned were the attributes of the item itself, and especially did not include usage needs.
        """,
        "solution_description": """
        To complete the recommendation, you need to generate an execution plan that can successfully generate the recommendation. 
        In order to complete the recommendation, you can directly search for products that meet the target attributes 
        in the product database through the keywords mentioned by the user.
        """,
        "thought_template": """
        Step 1: Determine the user's target products and attributes.
        Step 2: Use the item attributes to retrieve items in ItemRetriever.
        Step 3: Recommend the retrieved items to the user.
    """
    },
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
    },
    # "template_2": {
    #     "task_description": """
    #     In this conversational recommendation query, the user raises a specific demand without specifying
    #     a specific type of apparel, but instead asks for a set of clothes that can meet the demand to be recommended.
    #     """,
    #     "solution_description": """
    #     In this task, the user asks for a set of clothing recommendations, but does not specify a specific category.
    #     In order to recommend clothing that satisfies the user, we should first collect information to determine which categories of clothing to recommend,
    #     and then plan the recommendations for each category separately and give their own recommendation lists.
    #     """,
    #     "thought_template": """
    #     Phase 1: Collect information to determine which categories of clothing to recommend
    #     Step 1: Determine the user's needs.\n
    #     Step 2: Based on the need, obtain relevant knowledge through searcherAgent.\n
    #     Step 3: Based on the knowledge, determine about three clothing types to recommend and update plan.\n
    #     Phase 2: Provide recommendations for the clothing categories chosen.
    #     """
    # },
    "template_3": {
        "task_description": """
        In this conversational recommendation query, the user puts forward a specific requirement and 
        asks for a recommendation of a clothing of a specified type, and then additionally requires one or multiple clothing types to be matched with this cloth.
        """,
        "solution_description": """
        In this task, the user specify a specific clothing type, and ask for other types to be matched with this product.
        In order to recommend clothing that satisfies the user, the recommendation for the specified item should be obtained first, 
        and then the attributes of the matching clothing should be determined based on the demand and the attributes of the item.
        Based on these attributes, recommend these matching clothes.
        """,
        "thought_template": """
        Step 1: Determine the user's target product and needs.\n
        Step 2: Determine whether there are clear product attributes for this item retrieval. If not, you need to obtain relevant knowledge through searcherAgent.\n
        Step 3: Use the obtained item attributes to retrieve items in ItemRetriever.\n
        Step 4: The attributes of clothing for matching can be obtained through searcherAgent based on the recommended clothing attributes or the requirements in the query.\n
        Step 5: Use the obtained item attributes to retrieve clothing for matching in ItemRetriever.\n
        Step 6: Recommend each final list to the user.
        """
    },
    "template_4": {
        "task_description": """
        In this conversational recommendation query, the user puts forward a requirement,
        but is not clear the specific using scene or demand. And asks for a recommendation of a specific type of product.
        """,
        "solution_description": """
        In this task, the user puts forward a demand for clothing or beauty product recommendation, but expresses uncertainty about the demand,
        which means that more considerations need to be given to meet the needs of different possible specific scenarios.
        In order to recommend clothing or beauty product that satisfies users, we should first collect information and determine which
        categories the product attributes that meet user needs can be divided into (corresponding to different specific scenarios),
        and then plan recommendations for each specific situation and give their own recommendation lists.
        """,
        "thought_template": """
        Phase 1: Gather information to determine what categories the target needs can be classified into
        Step 1: Determine the user's needs.\n
        Step 2: Based on the need, obtain relevant knowledge through searcherAgent.\n
        Step 3: Based on the knowledge, divide the user need into about three specific scenarios. Only one recommend isn't enough. \n
        Phase 2: Provide recommendations for each specific scenario.
        """
    },
    "template_5": {
        "task_description": """
        In this conversational recommendation query, the user puts forward multiple usage scenarios requirements that 
        need to be met by only one item. 
        Then, the user requests recommendation for product of a specified type.
    """,
        "solution_description": """
        One product may not necessarily meet multiple complex needs.
        In order to give users satisfactory recommendations, it is necessary to determine the attributes corresponding 
        to the needs, filter them, and then make recommendations.
    """,
        "thought_template": """
        Step 1: Determine the user's needs.\n
        Step 2: Based on each need, obtain relevant knowledge through searcherAgent.\n
        Step 3: Based on the knowledge, only retain attributes that are not contradictory and potentially compatible with each other are retained. \n
        Step 4: Provide recommendations for the target.
    """
    },
    # "template_6": {
    #     "task_description": """
    #     In this conversational recommendation query, The user requested recommendations for a "set" of items and
    #     specifically specified one specific item to be included in the "set".
    #     """,
    #     "solution_description": """
    #     In this task, the user asks for a set of products recommendations, and specify a specific item.
    #     In order to recommend products that satisfies the user, the recommendation for the specified item should be obtained first,
    #     and then the types of products that can be a set with the specified product should be determined based on the attributes of the item,
    #     Based on these attributes, recommend these matching products.
    #     """,
    #     "thought_template": """
    #     Phase 1: Collect information to determine which types of products to recommend.
    #     Step 1: Determine the user's needs and specified product.\n
    #     Step 2: Use the item attributes to retrieve items in ItemRetriever.\n
    #     Step 3: Search for the product types that can be used with the specified product through searcherAgent.\n
    #     Step 4: Based on the knowledge, determine no more than two another product types
    #     (not include items that have already been recommended) to recommend and update plan.\n
    #     Phase 2: Provide recommendations for the matching product categories chosen.
    #     """
    # },
    # "template_7": {
    #     "task_description": """
    #     In this conversational recommendation query, the user asks for a recommendation of a product with specified attributes,
    #     and then requires one or multiple products to be matched with this product.
    #     """,
    #     "solution_description": """
    #     In this task, the user specify a specific product type, and ask for other products to be matched with this product.
    #     In order to recommend products that satisfies the user, the recommendation for the specified item should be obtained first,
    #     and then the attributes of the matching products should be determined based on the attributes of the item.
    #     Based on these attributes, recommend these matching products.
    #     """,
    #     "thought_template": """
    #     Step 1: Determine the user's target product.\n
    #     Step 2: Determine whether there are product attributes for this item retrieval.\n
    #     Step 3: Use the obtained item attributes to retrieve items in ItemRetriever.\n
    #     Step 4: The attributes of products for matching can be obtained through searcherAgent based on the recommended product attributes.\n
    #     Step 5: Use the obtained item attributes to retrieve product for matching in ItemRetriever.\n
    #     Step 6: Recommend each final list to the user.
    #     """
    # },

    "template_8": {
        "task_description": """
    In this conversational recommendation query, user puts forward one usage requirement and specifies one 
    beauty product type for recommendation (specification such as capacity or size is not a special usage requirement).
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
    },
    "template_9": {
        "task_description": """
    In this conversational recommendation query, the user raises a specific demand without specifying 
    a specific type of beauty product, but instead asks for a set of beauty products that can meet the demand to be recommended.
    """,
        "solution_description": """
    In this task, the user asks for a set of beauty product recommendations, but does not specify a specific category. 
    In order to recommend beauty product that satisfies the user, we should first collect information to determine which categories of beauty product to recommend, 
    and then plan the recommendations for each category separately and give their own recommendation lists.
    """,
        "thought_template": """
    Phase 1: Collect information to determine which categories of beauty product to recommend
    Step 1: Determine the user's needs.\n
    Step 2: Based on the need, obtain relevant knowledge through searcherAgent.\n
    Step 3: Based on the knowledge, determine about three beauty product types to recommend and update plan.\n
    Phase 2: Provide recommendations for the beauty product categories chosen. 
    """
    },
    "template_10": {
        "task_description": """
    In this conversational recommendation query, the user puts forward a specific requirement and 
    asks for a recommendation of a beauty product of a specified type, and then additionally requires one or multiple beauty product types to be matched with this product.
    """,
        "solution_description": """
    In this task, the user specify a specific beauty product type, and ask for other types to be matched with this product.
    In order to recommend beauty product that satisfies the user, the recommendation for the specified item should be obtained first, 
    and then the attributes of the matching beauty product should be determined based on the demand and the attributes of the item.
    Based on these attributes, recommend these matching beauty products.
    """,
        "thought_template": """
    Step 1: Determine the user's target product and needs.\n
    Step 2: Determine whether there are clear product attributes for this item retrieval. If not, you need to obtain relevant knowledge through searcherAgent.\n
    Step 3: Use the obtained item attributes to retrieve items in ItemRetriever.\n
    Step 4: The attributes of beauty product for matching can be obtained through searcherAgent based on the recommended beauty product attributes or the requirements in the query.\n
    Step 5: Use the obtained item attributes to retrieve beauty product for matching in ItemRetriever.\n
    Step 6: Recommend each final list to the user.
    """
    },
    "template_11": {
        "task_description": """
    In this conversational recommendation query, user puts forward one usage scene and specifies one music type for recommendation.
    """,
        "solution_description": """
    To complete the recommendation, you need to generate an execution plan that can successfully generate the recommendation. 
    In order to complete the recommendation, you need to search for musics that meet the target attributes 
    in the item database through keywords. However, for some complex user needs, the music attributes are unclear. 
    At this time, you need to call the searcher to search for relevant knowledge and obtain the music attributes 
    that can meet the user's needs.
    """,
        "thought_template": """
    Step 1: Determine the user's target music type and needs.\n
    Step 2: Determine whether there are clear music attributes for item retrieval. If not, you need to obtain relevant knowledge through searcherAgent.\n
    Step 3: Use the obtained music attributes to retrieve items in ItemRetriever.\n
    Step 4: Recommend the retrieved items to the user.\n
    """
    },
    "template_12": {
        "task_description": """
        In this conversational recommendation query, the user raises a usage requirement for music, and **in particular** ask for 'some different styles of music'.
    """,
        "solution_description": """
    In this task, users ask for recommendations of different styles of music for a purpose.
    In order to recommend music that satisfies users, we should first collect information and determine which styles of music to recommend.
    Then, for each category, we should plan recommendations for each purpose and give their own recommendation lists.
    """,
        "thought_template": """
    Phase 1: Collect information to determine which categories of music to recommend
    Step 1: Determine the user's needs.\n
    Step 2: Based on the need, obtain relevant knowledge through searcherAgent.\n
    Step 3: Based on the knowledge, determine about three music product types to recommend and update plan.\n
    Phase 2: Provide recommendations for the music categories chosen. 
    """
    },
    "template_13": {
        "task_description": """
    In this conversational recommendation query, the user puts forward a specific requirement scene and 
    asks for a recommendation of a music of a specified type, and then additionally requires one or multiple specific requirements for the same music type.
    """,
        "solution_description": """
    In this task, the user specifies a specific music type and puts forward multiple requirements.
    This may indicate that the user loves this music and needs to listen to or play it in different scenarios. 
    In order to recommend music that satisfies the user, we first need to determine the attributes of the music required 
    for each scenario. Then, recommend the same type of music with different attributes for each scenario.
    """,
        "thought_template": """
    Phase 1: Collect information to determine what music to recommend for each scene.
    Step 1: Determine the user's needs.\n
    Step 2: Based on the need, obtain relevant knowledge through searcherAgent.\n
    Step 3: Combine the specified music type to determine the specific music attributes of each scene.\n
    Phase 2: Provide recommendations for each scene. 
    """
    }
}
