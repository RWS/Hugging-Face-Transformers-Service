import json
from main import app  # Import the FastAPI app from main.py

def main():
    # Generate the OpenAPI schema
    openapi_schema = app.openapi()

    # Save the schema to a JSON file
    with open("openapi.json", "w", encoding="utf-8") as f:
        json.dump(openapi_schema, f, indent=2)

    print("OpenAPI schema has been saved to openapi.json")

if __name__ == "__main__":
    main()