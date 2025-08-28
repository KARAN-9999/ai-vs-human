### Explainability Summary

- **LIME** highlighted which individual words were most influential in predicting *AI* vs *Human*.  
  Example: words like "traditional", "education" pushed towards *Human*, while filler words ("the", "are") had weaker influence.  

- **Captum Integrated Gradients** revealed which **tokens in the Transformer embeddings** contributed most to the classification.  
  This allows us to interpret the model at a deeper embedding level.  

Together, these methods provide complementary insights:
- LIME → interpretable word-level feature importance.  
- Captum → gradient-based attribution at embedding/token level.  
