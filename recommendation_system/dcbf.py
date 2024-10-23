import numpy as np
import faiss
import json
import os

class DeepContentBasedFiltering:
    def __init__(self, model_config_path):
        """
        Khởi tạo lớp với đường dẫn đến tệp cấu hình model.json.
        :param model_config_path: Đường dẫn đến tệp model.json.
        """
        # Tải cấu hình model
        with open(model_config_path, 'r') as f:
            self.model_paths = json.load(f)
        
        # Chỉ giữ lại các model cho fashion và book
        self.model_paths = {k: v for k, v in self.model_paths.items() if k in ['fashion', 'book']}
        
        # Từ điển lưu trữ dữ liệu cho mỗi loại sản phẩm
        self.data = {}
        self.indices = {}
        
        # Tải dữ liệu và xây dựng chỉ mục cho từng loại sản phẩm (fashion và book)
        for product_type, model_path in self.model_paths.items():
            self.load_data(product_type, model_path)
            self.build_indices(product_type)
    
    def load_data(self, product_type, model_path):
        """
        Tải dữ liệu từ tệp .npy cho một loại sản phẩm.
        :param product_type: Loại sản phẩm ('fashion' hoặc 'book').
        :param model_path: Đường dẫn đến tệp .npy chứa dữ liệu.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Nạp dữ liệu từ tệp .npy
        data = np.load(model_path, allow_pickle=True).item()
        
        # Đảm bảo product_id luôn là kiểu Python int thay vì numpy int
        data['product_id'] = [int(pid) for pid in data['product_id']]  # Chuyển đổi từng giá trị sang Python int
        
        # Lưu dữ liệu vào từ điển
        self.data[product_type] = data
    
    def build_indices(self, product_type):
        """
        Xây dựng chỉ mục FAISS cho một loại sản phẩm.
        :param product_type: Loại sản phẩm ('fashion' hoặc 'book').
        """
        data = self.data[product_type]
        indices = {}
        
        # Tùy thuộc vào loại sản phẩm, xây dựng chỉ mục dựa trên vector tương ứng
        if product_type == 'fashion':
            # Chỉ mục cho vector hình ảnh
            vector_images = np.vstack(data['vector_image']).astype('float32')
            d_image = vector_images.shape[1]
            index_image = faiss.IndexFlatL2(d_image)
            index_image.add(vector_images)
            indices['image'] = index_image
            
            # Lưu trữ vector category và brand
            indices['category'] = np.vstack(data['vector_category'])
            indices['brand'] = np.vstack(data['vector_brand'])
        
        elif product_type == 'book':
            # Chỉ mục cho vector tên sản phẩm
            vector_names = np.vstack(data['vector_name']).astype('float32')
            d_name = vector_names.shape[1]
            index_name = faiss.IndexFlatL2(d_name)
            index_name.add(vector_names)
            indices['name'] = index_name
            
            # Lưu trữ vector category, author, publisher
            indices['category'] = np.vstack(data['vector_category'])
            indices['author'] = np.vstack(data['vector_author'])
            indices['publisher'] = np.vstack(data['vector_publisher'])
        
        else:
            raise ValueError(f"Unknown product type: {product_type}")
        
        # Lưu chỉ mục vào self.indices
        self.indices[product_type] = indices
    
    def recommend(self, product_id, product_type, top_k=10, exclude_viewed=True, viewed_product_ids=None):
        """
        Gợi ý sản phẩm tương tự dựa trên product_id và product_type.
        :param product_id: ID của sản phẩm cần tìm.
        :param product_type: Loại sản phẩm ('fashion' hoặc 'book').
        :param top_k: Số lượng sản phẩm gợi ý.
        :param exclude_viewed: Loại bỏ các sản phẩm đã xem hay không.
        :param viewed_product_ids: Danh sách các product_id đã xem.
        :return: Danh sách các product_id được gợi ý hoặc [] nếu product_id không hợp lệ.
        """
        data = self.data.get(product_type)
        if data is None:
            return []  # Return empty list if the product_type is not found

        indices = self.indices.get(product_type)
        if indices is None:
            return []  # Return empty list if the indices are not found

        # Kiểm tra xem product_id có trong dữ liệu không
        if product_id not in data['product_id']:
            return []  # Trả về danh sách rỗng nếu product_id không tồn tại

        # Lấy vị trí của product_id trong mảng
        idx = data['product_id'].index(product_id)

        if product_type == 'fashion':
            query_vector_image = data['vector_image'][idx].astype('float32')
            query_vector_category = data['vector_category'][idx]
            query_vector_brand = data['vector_brand'][idx]

            # Search using FAISS
            distances, indices_image = self.indices[product_type]['image'].search(np.array([query_vector_image]), top_k * 3)
            candidate_indices = indices_image[0]

            # Ensure indices are valid (within the bounds of all arrays)
            valid_candidate_indices = [i for i in candidate_indices if i < len(data['product_id'])]

            # Get the corresponding product IDs
            candidate_product_ids = [data['product_id'][i] for i in valid_candidate_indices]

            # Ensure all indices are valid for category and brand arrays
            valid_candidate_indices = [i for i in valid_candidate_indices 
                                    if i < len(self.indices[product_type]['category']) 
                                    and i < len(self.indices[product_type]['brand'])]

            candidate_categories = self.indices[product_type]['category'][valid_candidate_indices]
            category_distances = np.linalg.norm(candidate_categories - query_vector_category, axis=1)

            candidate_brands = self.indices[product_type]['brand'][valid_candidate_indices]
            brand_distances = np.linalg.norm(candidate_brands - query_vector_brand, axis=1)

            total_distances = distances[0][:len(valid_candidate_indices)] + category_distances + brand_distances
            recommendations = list(zip(candidate_product_ids, total_distances))

            # Loại bỏ chính sản phẩm đang xem
            #recommendations = [rec for rec in recommendations if rec[0] != product_id]

            # Loại bỏ các sản phẩm đã xem nếu cần
            if exclude_viewed and viewed_product_ids is not None:
                recommendations = [rec for rec in recommendations if rec[0] not in viewed_product_ids]
            if exclude_viewed:
                recommendations = [rec for rec in recommendations if rec[0] != product_id]

            recommendations.sort(key=lambda x: x[1])
            recommended_product_ids = [rec[0] for rec in recommendations[:top_k]]

        elif product_type == 'book':
            # Tương tự cho loại sản phẩm book
            query_vector_name = data['vector_name'][idx].astype('float32')
            query_vector_category = data['vector_category'][idx]
            query_vector_author = data['vector_author'][idx]
            query_vector_publisher = data['vector_publisher'][idx]

            distances, indices_name = self.indices[product_type]['name'].search(np.array([query_vector_name]), top_k * 3)
            candidate_indices = indices_name[0]

            # Ensure indices are valid (within the bounds of all arrays)
            valid_candidate_indices = [i for i in candidate_indices if i < len(data['product_id'])]

            candidate_product_ids = [data['product_id'][i] for i in valid_candidate_indices]

            # Ensure all indices are valid for category, author, and publisher arrays
            valid_candidate_indices = [i for i in valid_candidate_indices 
                                    if i < len(self.indices[product_type]['category']) 
                                    and i < len(self.indices[product_type]['author']) 
                                    and i < len(self.indices[product_type]['publisher'])]

            candidate_categories = self.indices[product_type]['category'][valid_candidate_indices]
            category_distances = np.linalg.norm(candidate_categories - query_vector_category, axis=1)

            candidate_authors = self.indices[product_type]['author'][valid_candidate_indices]
            author_distances = np.linalg.norm(candidate_authors - query_vector_author, axis=1)

            candidate_publishers = self.indices[product_type]['publisher'][valid_candidate_indices]
            publisher_distances = np.linalg.norm(candidate_publishers - query_vector_publisher, axis=1)

            total_distances = distances[0][:len(valid_candidate_indices)] + category_distances + author_distances + publisher_distances
            recommendations = list(zip(candidate_product_ids, total_distances))

            #recommendations = [rec for rec in recommendations if rec[0] != product_id]

            if exclude_viewed and viewed_product_ids is not None:
                recommendations = [rec for rec in recommendations if rec[0] not in viewed_product_ids]
            if exclude_viewed:
                recommendations = [rec for rec in recommendations if rec[0] != product_id]


            recommendations.sort(key=lambda x: x[1])
            recommended_product_ids = [rec[0] for rec in recommendations[:top_k]]

        else:
            return []  # Return empty list if the product_type is invalid

        return recommended_product_ids
