# app.py (Phiên bản 3.0 - Quay về Cấp độ 2 Ổn định)

import streamlit as st
from google import genai
from PIL import Image
import io

# --- Hàm phân tích lõi (Giữ nguyên) ---
def analyze_bio_image_streamlit(api_key, image_data, user_prompt, context_role):
    try:
        client = genai.Client(api_key=api_key) 
    except Exception as e:
        st.error(f"Lỗi xác thực API Key (Hình ảnh): {e}")
        return

    system_instruction = (
        f"Bạn là Trợ lý AI Sinh học THPT (BioScope AI). Hãy trả lời với vai trò là một chuyên gia {context_role}. "
        f"Phân tích hình ảnh, sử dụng kiến thức Sinh học THPT. Đưa ra nhận định chính xác và đặt câu hỏi/gợi ý phù hợp."
    )
    img = Image.open(image_data)
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[user_prompt, img],
        config={"system_instruction": system_instruction, "temperature": 0.5}
    )
    return response.text

# --- Cấu hình Trang & API Key ---
st.set_page_config(page_title="BioScope AI", layout="wide")

try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except KeyError:
    st.error("Lỗi: Không tìm thấy GEMINI_API_KEY. Vui lòng thêm vào Streamlit Secrets.")
    st.stop() 

# --- Giao diện Sidebar (Cấp độ 1) ---
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
st.sidebar.write("Trần Thụy Đông Hòa")
st.sidebar.write("Trường THPT Marie Curie")
st.sidebar.write("Email: hoattd@thptmariecuriehem.edu.vn")

# --- Tiêu đề chính ---
st.title("🔬 BioScope AI: Phân tích Hình ảnh Sinh học")
st.markdown("---")

# --- Giao diện chính (Cấp độ 2 - Chấm điểm) ---

# Vùng Upload ảnh
uploaded_file = st.file_uploader("1. Tải lên hình ảnh tiêu bản/thí nghiệm của bạn:", type=["png", "jpg", "jpeg"])

if uploaded_file:
    
    col_input, col_output = st.columns([2, 3]) 

    # ---- CỘT 1: INPUT ----
    with col_input:
        st.subheader("Bảng điều khiển")
        st.image(uploaded_file, caption=f"Ảnh đã tải lên: {uploaded_file.name}", use_column_width=True)
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
            
        prompt = st.text_area("3. Câu hỏi chi tiết của bạn:", default_prompt, height=150)
        
        request_scoring = st.checkbox("🔬 Yêu cầu AI chấm điểm hình ảnh (Thang 10)")
        
        submit_button = st.button("4. Phân tích bằng BioScope AI")

    # ---- CỘT 2: OUTPUT ----
    with col_output:
        st.subheader("Kết quả phân tích từ AI")
        
        if submit_button: 
            if prompt:
                with st.spinner('Đang phân tích hình ảnh bằng Gemini AI...'):
                    
                    final_prompt_to_ai = prompt 
                    
                    if request_scoring:
                        scoring_instruction = (
                            "\n\n--- YÊU CẦU CHẤM ĐIỂM ---"
                            "\nVới vai trò là một giáo viên Sinh học chuyên nghiệp, hãy chấm điểm hình ảnh này theo thang 10."
                            "\nTiêu chí chấm điểm: 1. Tính chính xác Sinh học (hiện tượng/cấu trúc). 2. Độ rõ nét, thẩm mỹ của tiêu bản/ảnh chụp. 3. Mức độ thành công của thí nghiệm (nếu có)."
                            "\nHãy trả lời theo cấu trúc sau:"
                            "\n**Điểm số:** [Điểm]/10"
                            "\n**Nhận xét chi tiết:** [Giải thích tại sao, chỉ rõ ưu điểm và nhược điểm cần cải thiện]"
                        )
                        final_prompt_to_ai += scoring_instruction 
                    
                    result = analyze_bio_image_streamlit(
                        GEMINI_API_KEY,
                        uploaded_file, 
                        final_prompt_to_ai, 
                        context
                    )

                    st.success("Phân tích Hoàn thành!")
                    with st.expander("Bấm vào đây để xem kết quả chi tiết", expanded=True):
                        st.markdown(result)
            else:
                st.error("Vui lòng nhập câu hỏi chi tiết.")
        else:
            st.info("Kết quả phân tích của AI sẽ xuất hiện tại đây sau khi bạn nhấn nút.")