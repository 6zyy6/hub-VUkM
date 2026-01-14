import pandas as pd
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from openai import OpenAI

def load_data():
    # 加载数据集
    dataset = pd.read_csv("dataset.csv", sep="\t", header=None, nrows=5000)
    texts = dataset[0].values
    labels = dataset[1].values
    print(f"数据量: {len(texts)}")
    print("标签分布:")
    print(pd.Series(labels).value_counts())
    return texts, labels

def preprocess(texts):
    # 文本预处理：jieba分词
    return [" ".join(jieba.lcut(str(text))) for text in texts]

def train_knn_model(texts, labels):
    # 训练KNN机器学习模型
    vectorizer = CountVectorizer(max_features=3000)
    features = vectorizer.fit_transform(texts)
    
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(features, labels)
    
    return vectorizer, knn

def predict_ml(text, vectorizer, model):
    # 机器学习模型预测
    processed_text = " ".join(jieba.lcut(text))
    text_features = vectorizer.transform([processed_text])
    prediction = model.predict(text_features)[0]
    return prediction

def predict_llm(text):
    # 大语言模型预测
    client = OpenAI(
        api_key="sk-62520a9927c44411999c01603112bd2b",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    
    completion = client.chat.completions.create(
        model="qwen-flash",
        messages=[
            {"role": "user", "content": f"""帮我进行文本分类：{text}

输出的类别只能从如下中进行选择，除了类别之外下列的类别，请给出最合适的类别。
FilmTele-Play            
Video-Play               
Music-Play              
Radio-Listen           
Alarm-Update        
Travel-Query        
HomeAppliance-Control  
Weather-Query          
Calendar-Query      
TVProgram-Play      
Audio-Play       
Other             
"""},
        ]
    )
    return completion.choices[0].message.content

def main():
    print("=== 文本分类双模型对比 ===\n")
    
    # 加载数据
    texts, labels = load_data()
    processed_texts = preprocess(texts)
    
    # 训练机器学习模型
    vectorizer, knn_model = train_knn_model(processed_texts, labels)
    
    # 测试样例
    test_texts = [
        "帮我导航到天安门",
        "播放一首周杰伦的歌曲", 
        "明天北京的天气怎么样",
        "设置明天早上7点的闹钟",
        "我想看最新的电影"
    ]
    
    print("\n=== 测试样例 ===")
    for text in test_texts:
        print(f"\n文本: {text}")
        
        # 机器学习预测
        ml_result = predict_ml(text, vectorizer, knn_model)
        print(f"机器学习(KNN): {ml_result}")
        
        # 大语言模型预测
        llm_result = predict_llm(text)
        print(f"大语言模型: {llm_result}")
        
        print("-" * 50)

if __name__ == "__main__":
    main()
