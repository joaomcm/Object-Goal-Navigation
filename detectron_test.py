from agents.utils.semantic_prediction import SemanticPredMaskRCNN
import pickle

args = pickle.load(open('./args_example.pkl','rb'))

pred = SemanticPredMaskRCNN(args)

