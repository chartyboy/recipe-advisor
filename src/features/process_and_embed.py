import os

from src import embeddings, process_recipes
from langchain.embeddings import HuggingFaceBgeEmbeddings

if __name__ == "__main__":
    # Process raw text data
    schema = ".recipe_name, .ingredients, [.instructions[].text]"
    columns = ["recipe_name", "ingredients", "instructions"]
    # fpath = [
    #     "./datasets/raw/epicurious.jl",
    #     "./datasets/raw/foodnetwork.jl",
    #     "./datasets/raw/allrecipes.jl",
    #     "./datasets/raw/tasty.jl",
    # ]
    # outpath = [
    #     "./datasets/interim/epicurious_cleaned.jsonl",
    #     "./datasets/interim/foodnetwork_cleaned.jsonl",
    #     "./datasets/interim/allrecipes_cleaned.jsonl",
    #     "./datasets/interim/tasty_cleaned.jsonl",
    # ]
    fpath = ["./datasets/raw/sample.jl"]
    outpath = ["./datasets/raw/processed_sample.jsonl"]

    rp = process_recipes.RecipeProcessor(schema)
    rp.process_recipes(dict(zip(fpath, outpath)))

    # Create embedding database
    model_name = "BAAI/bge-large-en"
    model_kwargs = {"device": "cuda"}
    encode_kwargs = {"normalize_embeddings": False}

    hf = HuggingFaceBgeEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )
    data_path = "./datasets/interim"
    persist_path = "./datasets/processed/sample_db"
    # sites = ["allrecipes.jl", "epicurious.jl", "foodnetwork.jl", "tasty.jl"]
    # sources = ["allrecipes.com", "epicurious.com", "foodnetwork.com", "tasty.co"]
    sites = ["processed_sample.jsonl"]
    sources = ["test.com"]
    json_path = [os.path.join(data_path, website) for website in sites]
    source_map = dict(zip(json_path, sources))
    base_collections = ["name", "ingredient", "instruction"]
    recipe_embed = embeddings.RecipeEmbeddings(
        json_path=json_path,
        embedding_model=hf,
        persist_path=persist_path,
        base_collections=base_collections,
        reset=False,
    )
    _ = recipe_embed.process()
    _ = recipe_embed.create_summed_collection(["name", "ingredient", "instruction"])

    print("Finished")
