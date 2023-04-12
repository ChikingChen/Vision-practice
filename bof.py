def get_image_features(kmeans, des):
    ## 使用 BoF 算法进行特征编码
    labels = kmeans.predict(des)
    bof = [0] * kmeans.n_clusters
    for i in range(kmeans.n_clusters): bof[i] = 0
    for i in labels: bof[i] += 1
    return bof