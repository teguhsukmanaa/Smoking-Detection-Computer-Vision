import streamlit as st

def app():
    # Libraries
    import pandas as pd
    import matplotlib.pyplot as plt


    # Streamlit option to turn off the disable warning
    st.set_option('deprecation.showPyplotGlobalUse', False)


    # Load data
    df = pd.read_csv('df_train_images_file.csv')


    # Page title
    st.title("Exploratory Data Analysis (EDA) of Smoking Images Prediction")


    # Data preview
    with st.container(border=True):
        st.subheader("Data Preview")
        data_ov = '''Berikut merupakan dataset yang digunakan sebagai training data untuk model training dalam project ini.'''
        st.markdown(data_ov)
        st.write(df.head())


    # EDA 1 - Class Data Proportion Pie Chart
    with st.container(border=True):
        # Subheader
        st.subheader("Class Data Proportion")

        # Pie Chart
        labels = ['Not-Smoking', 'Smoking']
        size = df['label'].value_counts()
        colors = ['skyblue', 'dodgerblue']
        explode = [0.1, 0]

        fig, axes = plt.subplots(figsize=(10, 5))
        plt.pie(size, colors = colors, explode = explode,
                labels = labels, startangle = 90, autopct = '%.2f%%')
        plt.title('Images Class Data Proportion', fontsize = 15)
        plt.legend()
        
        # Show figure
        st.pyplot()

        # Caption
        text_1 = '''Dapat diketahui bahwa proporsi data kelas pada dataset ini seimbang antara Smoking dan Not-Smoking dengan lebih spesifiknya yaitu terdiri dari 385 data gambar untuk masing-masing klasifikasinya. Sehingga tidak perlu mengkhawatirkan adanya penurunan performa akibat data imbalance.'''
        st.markdown(text_1)


    # EDA 2 - Image Size Comparison
    with st.container(border=True):
        # Subheader
        st.subheader("Image Size Comparison")

        # Calculate the average images size value
        average_images_sizes = df.groupby('label')['image_size_kb'].mean()

        plt.figure(figsize=(8, 6))
        average_images_sizes.plot(kind='bar', color=['dodgerblue', 'green'])

        # Add labels on top of each bar
        for index, value in enumerate(average_images_sizes):
            plt.text(index, value, f'{value:.2f}', ha='center', va='bottom')

        plt.title('Average Image Size Comparison of Smoking and Not-Smoking')
        plt.xlabel('Category')
        plt.ylabel('Average Image Size (KB)')
        plt.xticks(rotation=0)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Show figure
        st.pyplot()

        # Average all images size
        average_all_images_sizes = df['image_size_kb'].mean()
        st.write(f"Rerata keseluruhan ukuran file gambar: `{average_all_images_sizes}`")

        # Caption
        text_2 = '''Dapat diketahui bahwa data gambar untuk kategori Smoking memiliki rerata ukuran file yang sedikit lebih besar yaitu sekitar 74kb dibandingkan kategori Not-Smoking yaitu sekitar 69kb. Hal ini mungkin dikarenakan pada gambar katergori Smoking lebih banyak ditemukan detail seperti asap rokok. Kemudian, secar keseluruhan rerata ukuran file gambar yang ada pada dataset ini berkisar pada 72kb, adapun besar rerata ukuran file gambar tersebut memang tidak terlalu besar namun sepertinya dapat membuat waktu training cukup lama apabila tidak menggunakan GPU.'''
        st.markdown(text_2)