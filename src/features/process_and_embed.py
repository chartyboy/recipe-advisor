import os

from src import embeddings, process_recipes
from langchain.embeddings import HuggingFaceBgeEmbeddings


def embed():
    # Create embedding database
    # model_name = "BAAI/bge-large-en"
    model_name = "BAAI/bge-small-en-v1.5"
    model_kwargs = {"device": "cuda"}
    encode_kwargs = {"normalize_embeddings": False}

    hf = HuggingFaceBgeEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )
    data_path = "./datasets/interim"
    persist_path = "./datasets/processed/chroma_db_small"
    sites = [
        "allrecipes_cleaned.jsonl",
        "epicurious_cleaned.jsonl",
        "foodnetwork_cleaned.jsonl",
        "tasty_cleaned.jsonl",
    ]
    sources = ["allrecipes.com", "epicurious.com", "foodnetwork.com", "tasty.co"]
    # sites = ["processed_sample.jsonl"]
    # sources = ["test.com"]
    json_path = [os.path.join(data_path, website) for website in sites]
    source_map = dict(zip(json_path, sources))
    base_collections = ["name", "ingredient", "instruction"]
    recipe_embed = embeddings.RecipeEmbeddings(
        json_path=json_path,
        embedding_model=hf,
        persist_path=persist_path,
        base_collections=base_collections,
        reset=True,
    )
    _ = recipe_embed.process()
    _ = recipe_embed.create_summed_collection(["name", "ingredient", "instruction"])


def process():
    # Process raw text data
    schema = ".recipe_name, .ingredients, [.instructions[].text]"
    columns = ["recipe_name", "ingredients", "instructions"]
    fpath = [
        "./datasets/raw/epicurious.jl",
        "./datasets/raw/foodnetwork.jl",
        "./datasets/raw/allrecipes.jl",
        "./datasets/raw/tasty.jl",
    ]
    outpath = [
        "./datasets/interim/epicurious_cleaned.jsonl",
        "./datasets/interim/foodnetwork_cleaned.jsonl",
        "./datasets/interim/allrecipes_cleaned.jsonl",
        "./datasets/interim/tasty_cleaned.jsonl",
    ]
    # fpath = ["./datasets/raw/sample.jl"]
    # outpath = ["./datasets/interim/processed_sample2.jsonl"]

    rp = process_recipes.RecipeProcessor(schema)
    rp.process_recipes(dict(zip(fpath, outpath)), columns=columns)


if __name__ == "__main__":
    print("Cleaning data...")
    process()

    print("Now embedding...")
    embed()

    print("Finished")
