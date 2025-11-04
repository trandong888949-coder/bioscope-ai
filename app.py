# app.py (Phiên bản 4.0 - Cấp độ 3: Đã thêm RAG Hỏi-đáp PDF)

import streamlit as st
from google import genai
from PIL import Image
import io
import os

# --- Thư viện mới cho Cấp độ 3 (RAG) ---
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from PyPDF2 import PdfReader
# ----------------------------------------

# --- Hàm phân tích lõi (Hình ảnh) - Giữ nguyên ---
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

# --- Hàm mới cho Cấp độ 3 (Xử lý PDF) ---
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# --- Hàm mới cho Cấp độ 3 (Chia chunks) ---
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# --- Hàm mới cho Cấp độ 3 (Tạo Vector Store) ---
def get_vector_store(text_chunks, api_key):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        # Lưu vào session_state để tái sử dụng
        st.session_state.vector_store = vector_store
        st.sidebar.success("Đã xử lý xong tài liệu PDF!")
    except Exception as e:
        st.sidebar.error(f"Lỗi tạo Vector Store: {e}")

# --- Hàm mới cho Cấp độ 3 (Xử lý câu hỏi RAG) ---
def answer_pdf_question(api_key, user_question):
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key, temperature=0.3)
        chain = load_qa_chain(llm, chain_type="stuff")
        # Lấy vector_store từ session_state
        vector_store = st.session_state.vector_store
        docs = vector_store.similarity_search(user_question)
        response = chain.run(input_documents=docs, question=user_question)
        return response
    except Exception as e:
        return f"Lỗi khi trả lời câu hỏi: {e}"

# --- Cấu hình Trang & API Key ---
st.set_page_config(page_title="BioScope AI", layout="wide")

# Lấy API Key từ Secrets
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except KeyError:
    st.error("Lỗi: Không tìm thấy GEMINI_API_KEY. Vui lòng thêm vào Streamlit Secrets.")
    st.stop() # Dừng ứng dụng nếu không có API Key

# --- Giao diện Sidebar (Đã nâng cấp Cấp 3) ---
st.sidebar.title("🔬 Giới thiệu BioScope AI")
st.sidebar.info(
    """
    Đây là công cụ AI 2-trong-1: 
    1. **Phân tích Hình ảnh** (Tab chính).
    2. **Hỏi-đáp Tài liệu PDF** (Mới!)
    """
)
st.sidebar.markdown("---")

# --- Tính năng Cấp 3 (Tải PDF lên Sidebar) ---
st.sidebar.subheader("📚 Tính năng Hỏi-đáp Tài liệu")
st.sidebar.write("Tải lên file PDF (ví dụ: Sách giáo khoa, sách Campbell) để AI đọc và trả lời câu hỏi.")
pdf_docs = st.sidebar.file_uploader("Tải lên file PDF của bạn", accept_multiple_files=True, type="pdf")

if st.sidebar.button("Xử lý Tài liệu PDF"):
    if pdf_docs:
        with st.sidebar.spinner("Đang đọc và phân tích PDF... (Có thể mất vài phút)"):
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks, GEMINI_API_KEY)
    else:
        st.sidebar.warning("Vui lòng tải lên ít nhất một file PDF.")

st.sidebar.markdown("---")
st.sidebar.subheader("Thông tin tác giả")
st.sidebar.write("Trần Thụy Đông Hòa")
st.sidebar.write("Trường THPT Marie Curie")
st.sidebar.write("Email: hoattd@thptmariecuriehem.edu.vn")

# --- Giao diện chính (Chia làm 2 Tab) ---
st.title("🔬 BioScope AI: Trợ lý AI Sinh học THPT")

tab1, tab2 = st.tabs(["🖼️ Phân tích Hình ảnh (Cấp 2)", "📚 Hỏi-đáp Tài liệu (Cấp 3 MỚI)"])

# --- TAB 1: PHÂN TÍCH HÌNH ẢNH (Như Cấp 2) ---
with tab1:
    st.header("Chức năng Phân tích Hình ảnh & Chấm điểm")
    
    uploaded_file = st.file_uploader("1. Tải lên hình ảnh tiêu bản/thí nghiệm:", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        col_input, col_output = st.columns([2, 3]) 

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
            submit_button = st.button("4. Phân tích Hình ảnh")

        with col_output:
            st.subheader("Kết quả phân tích Hình ảnh")
            
            if submit_button: 
                if prompt:
                    with st.spinner('Đang phân tích hình ảnh bằng Gemini AI...'):
                        final_prompt_to_ai = prompt
                        if request_scoring:
                            scoring_instruction = (
                                "\n\n--- YÊU CẦU CHẤM ĐIỂM ---"
                                "\nVới vai trò là một giáo viên Sinh học chuyên nghiệp, hãy chấm điểm hình ảnh này theo thang 10."
                                "\nTiêu chí chấm điểm: 1. Tính chính xác Sinh học. 2. Độ rõ nét của ảnh chụp. 3. Mức độ thành công của thí nghiệm."
                                "\nHãy trả lời theo cấu trúc sau:"
                                "\n**Điểm số:** [Điểm]/10"
                                "\n**Nhận xét chi tiết:** [Giải thích tại sao, chỉ rõ ưu điểm và nhược điểm]"
                            )
                            final_prompt_to_ai += scoring_instruction
                        
                        result = analyze_bio_image_streamlit(GEMINI_API_KEY, uploaded_file, final_prompt_to_ai, context)
                        st.success("Phân tích Hoàn thành!")
                        with st.expander("Bấm vào đây để xem kết quả chi tiết", expanded=True):
                            st.markdown(result)
                else:
                    st.error("Vui lòng nhập câu hỏi chi tiết.")
            else:
                st.info("Kết quả phân tích của AI sẽ xuất hiện tại đây sau khi bạn nhấn nút.")

# --- TAB 2: HỎI-ĐÁP TÀI LIỆU (Cấp 3 MỚI) ---
with tab2:
    st.header("Chức năng Hỏi-đáp dựa trên Tài liệu PDF")
    st.info("Vui lòng tải lên và xử lý file PDF ở thanh Sidebar bên trái trước khi đặt câu hỏi.")
    
    # Khởi tạo session_state cho lịch sử chat
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Hiển thị lịch sử chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Ô nhập câu hỏi của người dùng
    user_question = st.chat_input("Đặt câu hỏi về tài liệu PDF bạn đã tải lên...")

    if user_question:
        # Hiển thị câu hỏi của người dùng
        with st.chat_message("user"):
            st.markdown(user_question)
        st.session_state.messages.append({"role": "user", "content": user_question})
        
        # Kiểm tra xem vector store đã sẵn sàng chưa
        if "vector_store" not in st.session_state:
            st.warning("Vui lòng tải lên và 'Xử lý Tài liệu PDF' ở Sidebar trước khi đặt câu hỏi.")
        else:
            # Lấy câu trả lời từ AI
            with st.spinner("AI đang tìm kiếm trong tài liệu..."):
                response = answer_pdf_question(GEMINI_API_KEY, user_question)
                
                # Hiển thị câu trả lời của AI
                with st.chat_message("assistant"):
                    st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})