from whoosh.fields import Schema, TEXT, ID

def get_schema():
    return Schema(
        product_id=ID(stored=True, unique=True),
        name=TEXT(stored=True),
        description=TEXT,
        authors=TEXT,
        category=TEXT
    )