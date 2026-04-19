def get_top_words(tfidf, model, vec):
    feature_names = tfidf.get_feature_names_out()
    coefs = model.coef_[0]

    vec_array = vec.toarray()[0]
    indices = vec_array.nonzero()[0]

    word_scores = [(feature_names[i], coefs[i]) for i in indices]
    word_scores = sorted(word_scores, key=lambda x: abs(x[1]), reverse=True)[:10]

    return word_scores
