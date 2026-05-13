import joblib
from model.lda import MultiHeadLDA

# load the broken pickle
lda = joblib.load("model/mh_lda.pkl")

# re-save it with correct module path
joblib.dump(lda, "model/mh_lda_fixed.pkl")

print("✅ Fixed LDA pickle saved as mh_lda_fixed.pkl")
