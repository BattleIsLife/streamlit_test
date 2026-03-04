import streamlit as st
import pickle
import random
import numpy as np

# File
pkl = pickle.load(open('random_forest_model_simple.pkl', 'rb'))

st.title("Dự đoán xem bạn có dễ bị thôi học không")
age = st.slider("Tuổi", 18, None, 18)

# Các input khác

tuition_debt = st.slider("Học phí còn nợ (VND)", 0, 100_000_000, 0, step=500_000)

training_score_mixed = st.slider("Điểm rèn luyện trung bình", 0, 100, 80)



# ----------------- Input 40 điểm attendance -----------------
st.subheader("Điểm chuyên cần / assignment (40 môn/kỳ)")

score_input_type = st.selectbox("Cách nhập điểm", ["Thủ công", "Random"])

att_score = []

if score_input_type == "Random":
    att_score = [random.randint(-1, 10) for _ in range(40)]
    count_f = sum(1 for s in att_score if s < 5)
    st.write(f"Số môn dưới 5 điểm (F): **{count_f}**")
    with st.expander("Xem chi tiết điểm random"):
        for i, score in enumerate(att_score, 1):
            st.write(f"Môn {i}: {score}")

else:  # Thủ công
    st.info("Nhập thủ công 40 điểm (có thể dùng -1 nếu chưa có điểm)")
    cols = st.columns(4)  # Chia 4 cột cho gọn
    temp_scores = []
    for i in range(40):
        col = cols[i % 4]
        with col:
            score = st.number_input(f"Môn {i+1}", -1, 10, 7, key=f"score_{i}")
            temp_scores.append(score)
    att_score = temp_scores
    count_f = sum(1 for s in att_score if s < 5)
    st.write(f"Số môn dưới 5 điểm (F): **{count_f}**")

if st.button("Dự đoán tình trạng học tập", type="primary"):
    data = [
        age,
        tuition_debt,
        count_f,                           # số môn F
        training_score_mixed
    ]

    # Thêm 40 điểm attendance vào cuối
    data.extend(att_score)

    # Chuyển thành numpy array 2D (1 mẫu)
    input_data = np.array([data])

    try:
        prediction = pkl.predict(input_data)[0]
        prob = pkl.predict_proba(input_data)[0]

        st.divider()
        st.subheader("Kết quả dự đoán")

        labels = {0: "Bình thường", 1: "Cảnh báo học vụ", 2: "Nguy cơ thôi học"}
        st.metric("Tình trạng dự đoán", labels.get(prediction, "Không xác định"))

        if prediction == 0:
            st.success("Bạn đang học tập ổn định!")
        elif prediction == 1:
            st.warning("Cần chú ý cải thiện điểm số và tham gia hoạt động hơn.")
        elif prediction == 2:
            st.error("Có nguy cơ cao bị thôi học. Hãy liên hệ cố vấn học tập ngay!")

    except Exception as e:
        st.error(f"Lỗi khi dự đoán: {e}")
