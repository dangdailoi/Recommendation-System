import os
from whoosh.index import create_in, open_dir
from whoosh.qparser import MultifieldParser
from db.dbo import Product
from .schema import get_schema

class TextSearch:
    def __init__(self, index_dir="indexdir"):
        self.index_dir = index_dir
        if not os.path.exists(self.index_dir):
            os.mkdir(self.index_dir)
        self.schema = get_schema()

    def create_index(self):
        if not os.path.exists(os.path.join(self.index_dir, "MAIN")):
            ix = create_in(self.index_dir, self.schema)
            writer = ix.writer()

            products = Product.query.all()
            for product in products:
                authors = ", ".join([a.author_name for a in product.authors]) if product.authors else ''
                categories = ", ".join([c.category_name for c in product.categories]) if product.categories else ''

                writer.add_document(
                    product_id=str(product.product_id),
                    name=product.product_name,
                    description=product.product_description or '',
                    authors=authors,
                    category=categories
                )
            writer.commit()

    def search(self, query_str, limit=100):
        if not os.path.exists(self.index_dir):
            self.create_index()

        ix = open_dir(self.index_dir)
        query_str = query_str.strip()

        with ix.searcher() as searcher:
            parser = MultifieldParser(["name", "description", "authors", "category"], ix.schema)
            query = parser.parse(query_str)
            results = searcher.search(query, limit=limit)

            search_results = []
            for result in results:
                search_results.append((
                    result['product_id'],
                    result['name'],
                    result.get('description', ''),
                    result.get('authors', ''),
                    result.get('category', '')
                ))

        return search_results