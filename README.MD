# Hệ Thống Gợi Ý Sản Phẩm

## Giới thiệu
Trong thời đại số hóa hiện nay, việc tối ưu hóa trải nghiệm người dùng là một yếu tố then chốt trong các hệ thống thương mại điện tử. Người dùng không chỉ mong đợi các sản phẩm chất lượng mà còn cần sự hỗ trợ từ hệ thống để tìm kiếm sản phẩm phù hợp một cách nhanh chóng và hiệu quả. 

Các hệ thống gợi ý sản phẩm đã phát triển mạnh mẽ để cá nhân hóa trải nghiệm mua sắm. Học sâu (Deep Learning) và học tăng cường (Reinforcement Learning) là hai lĩnh vực quan trọng trong trí tuệ nhân tạo, góp phần đáng kể trong việc nâng cao khả năng dự đoán và cá nhân hóa của các mô hình gợi ý, từ đó cải thiện trải nghiệm người dùng.

## Mục tiêu
Đề tài này tập trung vào việc tối ưu hóa trải nghiệm người dùng thông qua hệ thống gợi ý sản phẩm sử dụng học sâu và học tăng cường:
- Học sâu được áp dụng để phân tích các dữ liệu phức tạp như hình ảnh, văn bản, và lịch sử mua hàng.
- Học tăng cường giúp tối ưu hóa các quyết định dựa trên phản hồi của người dùng theo thời gian thực.
- Phát triển các giải pháp gợi ý sản phẩm tiên tiến nhằm cá nhân hóa tương tác của người dùng trong môi trường thương mại điện tử.

## Kiến trúc dự án
Dự án bao gồm các thành phần chính như sau:
- `app.py`: File chính khởi chạy ứng dụng Flask.
- `db/`: Thư mục chứa các file liên quan đến cơ sở dữ liệu.
    - `dbo.py`: Định nghĩa các lớp ORM sử dụng SQLAlchemy.
    - `dbo.sql`: Các câu lệnh SQL cho cơ sở dữ liệu.
- `model/`: Thư mục chứa các mô hình học máy.
    - `Content-based/`: Mô hình lọc nội dung.
        - Mô hình về thông tin sản phẩm, dựa trên đặc trưng sản phẩm để tạo gợi ý.
    - `Image/`: Mô hình tìm kiếm hình ảnh.
        - Mô hình này cho phép tìm kiếm sản phẩm dựa trên đặc trưng hình ảnh.
    - `Reference-Learning/`: Mô hình học tham khảo.
        - Mô hình này để cá nhân hóa người dùng.
    - **Lưu ý**: Các mô hình học sâu có dung lượng lớn không được đính kèm trong dự án. Người dùng cần tự tạo và lưu vào thư mục tương ứng (`npy`, `h5`).
- `recommendation_system/`: Chứa các mô hình gợi ý.
    - `dcbf.py`: Mô hình lọc cộng tác dựa trên nội dung sâu.
    - `dqn.py`: Mô hình học tăng cường sử dụng Deep Q-Learning.
- `search_engine/`: Thư mục cho công cụ tìm kiếm sản phẩm.
    - `image_search.py`: Tìm kiếm sản phẩm qua hình ảnh.
    - `text_search.py`: Tìm kiếm sản phẩm qua văn bản.
    - `schema.py`: Định nghĩa schema cho Whoosh.
- `static/` và `templates/`: Thư mục chứa các file tĩnh và giao diện.

## Hướng dẫn cài đặt
1. **Yêu cầu hệ thống**:
   - Python >= 3.8
   - Các thư viện yêu cầu (có trong `requirements.txt`)

2. **Cài đặt thư viện**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Thiết lập cơ sở dữ liệu**:
   - Tạo cơ sở dữ liệu và chạy các file SQL trong `db/dbo.sql` nếu cần, lưu ý dữ liệu bạn có thể tự tạo hoặc crawl về.

4. **Chạy ứng dụng**:
   ```bash
   python app.py
   ```
   Ứng dụng sẽ chạy tại `http://localhost:5000`.

## Cấu trúc dữ liệu và mô hình
- **Mô hình học sâu**: Sử dụng mạng nơ-ron tích chập (CNN) để trích xuất đặc trưng từ hình ảnh sản phẩm.
- **Học tăng cường**: Áp dụng Deep Q-Learning để học từ phản hồi của người dùng.
- **Mô hình lọc nội dung**: Sử dụng đặc trưng được trích xuất từ các mô hình học sâu để đề xuất sản phẩm.

## Đánh giá hệ thống
- Hiệu suất hệ thống được đo lường qua các chỉ số như:
  - Tốc độ phản hồi
  - Mức độ hài lòng của người dùng

## Thư viện sử dụng
- `Flask` - Framework web cho Python.
- `SQLAlchemy` - Thư viện ORM cho việc quản lý cơ sở dữ liệu.
- `tensorflow` - Thư viện học sâu.
- `faiss` - Thư viện tìm kiếm hiệu suất cao.
- `whoosh` - Công cụ tìm kiếm toàn văn.

## Liên hệ
- **Tác giả**: Đặng Đại Lợi
- **Email**: dailoi.ddl@gmail.com
- **Website**: [\[Website (trước đây)\]](https://dailoi-ddl.glitch.me/)