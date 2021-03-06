# Decision Tree ID3 - Julia
Decision Tree ID3 algorithm

Cài đặt thuật toán cây quyết định ID3 cho việc phân lớp tập dữ liệu hoa Iris ([Iris Dataset](https://www.kaggle.com/uciml/iris))

Ngôn ngữ: Julia

# Iris Dataset
Tập dữ liệu bao gồm các cột:
- Id
- SepalLengthCm
- SepalWidthCm
- PetalLengthCm
- PetalWidthCm
- Species

# Các bước thực hiện
- Chia tập dữ liệu ngẫu nhiên thành 2 tập training và test tương ứng theo tỷ lệ 2/3 và 1/3.
- Cài đặt thuật toán cây quyết định dựa trên Entropy.
- Do các thuộc tính của tập iris đều có giá trị liên tục, ta cần rời rạc hóa từng thuộc tính bằng cách chọn một trong các giá trị của thuộc tính làm ngưỡng cutoff để chia các giá trị thuộc tính thành 2 phần sao cho Entropy là thấp nhất.
- Kết quả phân tích sử dụng độ đo accuracy trên tập test.
