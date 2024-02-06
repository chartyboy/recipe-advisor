"""
Collection of LLM prompts pre-formatted and instatiated as LangChain templates.

Template Constants
------------------
DEFAULT_DOCUMENT_PROMPT : str
    LangChain text template to convert LangChain Documents into template strings

beef_recipe : str
    Example of a beef stroganoff recipe.

mushroom_recipe : str
    Example of a mushroom stroganoff recipe.

modified_name_prompt : str
    Prompt template for altering a recipe name based on provided recipe changes.

cot_multi_prompt : str
    Prompt template for incorporating chain-of-thought and retrieval-augmented 
    generation prompt engineering.

strip_name_prompt : str
    Prompt template for finding the recipe name in a conversational response.
"""

from langchain.prompts import ChatPromptTemplate, PromptTemplate

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")

beef_recipe = """Recipe Name: Beef Stroganoff, 
Ingredients: 
1 pound ground beef, 
0.5 cup chopped onion, 
1 tablespoon all-purpose flour, 
0.5 teaspoon salt, 
0.25 teaspoon paprika, 
1 (10.75 ounce) can condensed cream of mushroom soup, 
1 cup sour cream, 
8 ounces egg noodles, 

Cooking Instructions: 
1. In a large skillet over medium heat, sauté beef and onions for 10 minutes, or until meat is browned and onion is tender.
2. Stir in flour, salt, and paprika. Add condensed soup, mix well, and cook, uncovered, for 20 minutes.
3. Reduce heat to low and add sour cream, stirring well and allowing to heat through. Cover and set this mixture aside.
4. Cook egg noodles according to package directions. Drain. Serve beef mixture over noodles., """

mushroom_recipe = """Recipe Name: Portobello Mushroom Stroganoff, 
Ingredients: 
0.75 pound portobello mushrooms, 
0.5 cup chopped onion, 
1 tablespoon all-purpose flour, 
0.5 teaspoon salt, 
0.25 teaspoon paprika, 
1 (10.75 ounce) can condensed cream of mushroom soup, 
1 cup sour cream, 
8 ounces egg noodles, 

Cooking Instructions: 
1. In a large skillet over medium heat, sauté mushroom and onions for 10 minutes, or until meat is browned and onion is tender.
2. Stir in flour, salt, and paprika. Add condensed soup, mix well, and cook, uncovered, for 20 minutes.
3. Reduce heat to low and add sour cream, stirring well and allowing to heat through. Cover and set this mixture aside.
4. Cook egg noodles according to package directions. Drain. Serve mushroom mixture over noodles., """

BASE_CONTEXT_INPUTS = {"beef_recipe": beef_recipe, "mushroom_recipe": mushroom_recipe}

modified_name_prompt = ChatPromptTemplate.from_template(
    """
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible. Your job is to come up with new names for recipes. You will 
be given an original recipe and a request from the user to change the name of the recipe to a more fitting one.

### Instruction:
Q. Complete the following task by reasoning step-by-step. Create a new recipe name for 'Beef Stroganoff' where beef will be replaced with portobello mushrooms,
and vegetable oil will be replaced with canola oil. Return the name of the recipe at the end of your response.
{beef_recipe}

A. Each modification to the recipe should be reflected in the name of the new recipe. The original recipe name is "Beef Stroganoff" because it contains beef. Since portobello mushrooms
will be substituted in place of beef, 'Beef' should be removed from the recipe's name as it will no longer be in the new recipe.
The word 'Beef' in the title will need to be replaced with the new ingredient, which is 'Portobello Mushroom'. Vegetable oil is listed as an ingredient in the original recipe,
but it is not mentioned in the recipe name. The replacement, canola oil, should be mentioned in the name of the new recipe to indicate that this ingredient was changed. Thus, the new name for the modified recipe
should be 'Portobello Mushroom Stroganoff with Canola Oil'. Final Answer: Portobello Mushroom Stroganoff with Canola Oil

### Input:
Q. Complete the following task by reasoning step-by-step. Create a new recipe name for {recipe_name} where in the new recipe, {customization}.
Return the name of the recipe at the end of your response.
{user_recipe}
### Response:
A.
"""
).partial(beef_recipe=BASE_CONTEXT_INPUTS["beef_recipe"])

cot_multi_prompt = ChatPromptTemplate.from_template(
    """
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible. Your job is to create a new recipe based on the user's instructions. You will 
be given an original recipe and a request from the user to customize the recipe.To help you customize the recipe, several example recipes will be provided based on the user's request.


### Instruction:
Q. Complete the following task by reasoning step-by-step. Create a new recipe called Mushroom Stroganoff from this recipe for Beef Stroganoff. In the new recipe, replace the ground beef from the original recipe with portobello mushrooms.
Original Recipe:
{beef_recipe}

A. The request to replace ground beef for portobello mushrooms indicates that any references to ground beef should be changed to use portobello mushrooms instead. 
In the ingredient list, ground beef is mentioned as an ingredient for the original recipe, which should now be replaced with portobello mushrooms instead. 
In addition, the cooking steps will need to be modified reflect the change in ingredients.
Steps 1 and 4 directly mention ground beef, so these steps will need to be updated. Step 1 should say to "sauté mushroom and onions" instead of "sauté beef and onions".
Step 4 should be changed to mention the mushroom mixture instead of the beef mixture. The modified recipe will look like this:
{mushroom_recipe}

### Input:

Example Recipes:
{retrieved_recipes}

Q. Complete the following task by reasoning step-by-step. Create a new recipe called {new_name} from this recipe for {recipe_name}. In the new recipe, {customization}.
Original Recipe:
{user_recipe}

### Response:
A. 
"""
).partial(**BASE_CONTEXT_INPUTS)

strip_name_prompt = ChatPromptTemplate.from_template(
    """
    You are a helpful, respectful and honest assistant. Always answer as helpfully as possible. Your task is to find the final name of the recipe in a group of sentences. Return the name of the recipe and nothing else.

    ### Input:
    Find the final name of the recipe in the following sentences. Return only the name of the recipe.
    {resp}
    
    ### Response:
"""
)
