# app.py (Phiên bản 2.0 - Dùng Streamlit Secrets)

import streamlit as st
from google import genai
from PIL import Image
import io

# --- Hàm phân tích lõi (Đã sửa để dùng Secrets) ---
def analyze_bio_image_streamlit(image_data, user_prompt, context_role):
    
    # 1. Lấy API Key từ Streamlit Secrets (Bảo mật)
    try:
        # Lấy key từ mục "Secrets" của Streamlit Cloud
        api_key = st.secrets["GEMINI_API_KEY"]
        
        # Kiểm tra xem key có trống không
        if not api_key:
            # Dòng này sẽ chỉ hiển thị khi deploy lên Streamlit Cloud
            st.error("Lỗi: GEMINI_API_KEY chưa được thiết lập trong Streamlit Secrets.")
            return

        # Khởi tạo client với API Key
        client = genai.Client(api_key=api_key) 
    
    except KeyError:
        st.error("Lỗi: Không tìm thấy GEMINI_API_KEY. Vui lòng thêm vào Streamlit Secrets.")
        return
    except Exception as e:
        st.error(f"Lỗi xác thực API Key: {e}")
        return

    # 2. Xây dựng Prompt (Giữ nguyên)
    system_instruction = (
        f"Bạn là Trợ lý AI Sinh học THPT (BioScope AI). Hãy trả lời với vai trò là một chuyên gia {context_role}. "
        f"Phân tích hình ảnh, sử dụng kiến thức Sinh học THPT. Đưa ra nhận định chính xác và đặt câu hỏi/gợi ý phù hợp."
    )

    img = Image.open(image_data)

    # 3. Gọi Gemini API (Sử dụng model Flash theo đề xuất)
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[user_prompt, img],
        config={"system_instruction": system_instruction, "temperature": 0.5}
    )
    return response.text

# --- Cấu hình Giao diện Streamlit (Giữ nguyên như cũ) ---
st.set_page_config(page_title="BioScope AI", layout="wide")
# --- BẮT ĐẦU CODE MỚI THÊM VÀO (Sidebar) ---
st.sidebar.title("🔬 Giới thiệu BioScope AI")
st.sidebar.info(
    """
    Đây là công cụ ứng dụng Gemini AI để phân tích hình ảnh Sinh học THPT. 
    Ứng dụng này giúp học sinh tự học, tự kiểm tra kiến thức và hỗ trợ 
    giáo viên trong công tác giảng dạy.
    """
)
st.sidebar.markdown("---")
st.sidebar.subheader("Thông tin tác giả")
st.sidebar.write("Trần Thụy Đông Hòa") # Bạn có thể sửa lại tên nếu muốn
st.sidebar.write("Trường THPT Marie Curie")
st.sidebar.write("Email: hoattd@thptmariecuriehem.edu.vn")
# --- KẾT THÚC CODE MỚI ---
st.title("🔬 BioScope AI: Phân tích Hình ảnh Sinh học")
st.markdown("---")

# Vùng Upload ảnh
uploaded_file = st.file_uploader("1. Tải lên hình ảnh tiêu bản/thí nghiệm của bạn:", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Hiển thị ảnh
    st.image(uploaded_file, caption=f"Ảnh đã tải lên: {uploaded_file.name}", width=300)

    # Chọn vai trò và nhập Prompt
    role = st.radio(
        "2. Chọn Vai trò (Tối ưu hóa phản hồi AI):",
        ("Học sinh (Tự học & Kiểm tra)", "Giáo viên (Kiểm tra & Tạo tư liệu)")
    )
    
    if role == "Học sinh (Tự học & Kiểm tra)":
        context = "Học sinh tự học"
        default_prompt = "Đây có phải là tiêu bản/thí nghiệm đúng không? Hãy giải thích hiện tượng và đặt cho tôi 2 câu hỏi ôn tập."
    else:
        context = "Giáo viên chuyên môn"
        default_prompt = "Đánh giá tính chính xác. Nếu đúng, gợi ý một hoạt động tiếp theo. Nếu sai, giải thích lỗi sai sinh học cơ bản."
        
    prompt = st.text_area("3. Câu hỏi chi tiết của bạn:", default_prompt)
    
    # Nút Submit
    if st.button("4. Phân tích bằng BioScope AI"):
        if prompt:
            with st.spinner('Đang phân tích hình ảnh bằng Gemini AI...'):
                result = analyze_bio_image_streamlit(uploaded_file, prompt, context)
                st.success("Phân tích Hoàn thành!")
                st.markdown("### Kết Quả Phân Tích từ AI:")
                st.info(result)
        else:
            st.error("Vui lòng nhập câu hỏi chi tiết.")