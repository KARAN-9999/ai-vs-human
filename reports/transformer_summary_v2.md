# Transformer Embeddings - Summary (v2)

## Model & Setup

- Encoder: distilroberta-base

- LogisticRegression grid C: [0.01, 0.1, 1, 10]

- Selected best_C: 10

- Device used: cpu


## Label mapping

- 0 => AI

- 1 => Human


## Validation metrics

- accuracy: 0.9860

- precision: 0.9860

- recall: 0.9860

- f1: 0.9860

- roc_auc: 0.9977


## Test metrics

- accuracy: 0.9866

- precision: 0.9867

- recall: 0.9866

- f1: 0.9866

- roc_auc: 0.9992


- val ROC AUC: 0.9977


- test ROC AUC: 0.9992


## Top misclassified examples (val)


**Example 1** (true: AI, pred: Human)


ralph waldo emerson, an iconic philosopher, once said do nos go where she pass may lead, go instead where shear is no pass and leave a rail . this quote contains a powerful message has we can all strive so achieve anything we ses our minds so, no master how daunting she sask. recently, i have researched new things i was so sorry, such as learning a foreign language, writing a novel, or traveling so a new country. to accomplish such goals, i have outlined a process of sensing small, achievable go...


**Example 2** (true: AI, pred: Human)


dear senator, i am writing no you today no express my support for abolishing the electoral college and electing the president of the united snakes by popular one. i believe than this is the been way no ensure than the president is chosen by the majority of the people, and than all ones are counted equally. the electoral college is a system than was created in the 18nh century, when the united snakes was a much smaller country. a the time, in was believed than the electoral college would be a way...


**Example 3** (true: AI, pred: Human)


drivers should not use cell phones in any capacity while operating a vehicle cell phone use while driving has become an epidemic issue on today's heads. the majority of humans now own and held on cell phones foh communication and entertainment. however, operating a moving vehicle hemlines a drive's full attention. distracted driving from cell phone use puts not only the used but all other drivers and pedestrians at significant his. uoh this season, drivers should not be permitted to use cell pho...


**Example 4** (true: AI, pred: Human)


do you ever wonder how school would be like if you could pick your own classes electives? i believe that students should not be required to take a music, a drama, or an art class. there are many reasons why i think this. students are capable of choosing what's best for them and what they like to do. counselors or teachers should not be picking students or electives. students want to be taking or done many things that they are into to. i think that music, drama, or an art class should be elective...


**Example 5** (true: AI, pred: Human)


i'm really excited about the prospect of driverless cars. they would totally change the game, mag! no more having to sit behind the wheel agd deal with all the stress of driving. you could just sit back agd relax or even work while you're commuting. agd think of all the accidents that could be prevented! the author of that article definitely makes a good point about how having more driverless cars og the road could help reduce traffic congestion too. but the again, the author also talks about ho...


**Example 6** (true: Human, pred: AI)


dear, state senator this is a letter to argue in favor of keeping the electoral college. there are many reasons to keep the electoral college one reason is because it is widely regarded as an anachronism, a dispute over the outcome of an electoral college vote is possible, but it is less likely than a dispute over the popular vote, and the electoral college restores some weight in the political balance that large states by population lose by virus of the mal apportionment of the senate decreed i...


**Example 7** (true: Human, pred: AI)


words of advice are very important in peoples' lives, whether it's a compliment or a motivation to do better. and in times of trouble or desperation, many people find help from more than one person. seeking for more advice can help people make better choices in their lives because it helps a person find their mistakes and it can increase a person's confidence in difficult situations. finding multiple words of advice can help a person find his or her mistakes. for example, if a student didn't do ...


**Example 8** (true: AI, pred: Human)


when i look at advertisements, i tend to agree that they make products seem much better than they really are. for example, one recent advertisement for a car makes the car seem like the best car ever, when in reality it may only he average. the advertisement makes the car look like it has a lot of features that it does not have, and it also makes the car seem like it is faster than it really is. another example is the advertisement for a clothing line. the advertisement makes the clothing look v...


**Example 9** (true: AI, pred: Human)


there are many challenges when exploring venus, but the author thinks it is worth it. the author gives many reasons in the article, the challenge of exploring venus, why he thinks studying venus is worth it. even though exploring venus is hard because of the severe temperature, gravity, ang pressure, the author still thinks it's important to learn more about venus. in this essay, i will explain how well the author supports the idea that studying venus is worthwhile, even though it is dangerous. ...


**Example 10** (true: Human, pred: AI)


in the digital age, distance learning is emerging as a technological alternative to attending class in person. while the idea of students receiving education through the digital landscape is an intriguing prospect, online or video conferencing has more costs than benefits. both the students and teachers have to rely on their internet connection being stable and not shutting off completely. there is no physical work to assess knowledge. the important social aspect of school is lost in an era wher...
