import chromadb


def main():
    client = chromadb.HttpClient(host="database", port=8000)
    print(client.list_collections())


if __name__ == "__main__":
    main()
