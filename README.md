# RL-Network-Recommendations-Project

# Phase 1

For the first phase of the project : We created an algorithm that learns to recommend "tuples" of items in video applications (like YouTube), where a user consumes multiple videos during a "session" before ending the session/episode.
Rules : 
1) We have K content items. For every pair of items, i and j, a random value u_ij in [0,1] was created, that indicates how related content j is to content i. This i basically a random array U of size K-by-K. 
We assumed the diagonal of this array (i.e. all elements u_ii) equal to 0 (to avoid recommending the same item just watched). Τhere is a threshold u_min in [0,1] below which two contents are "irrelevant".

2)  C out of these contents are cached (C <= 0.2 K). Cached items have cost 0, and non-cached have cost 1.

About the User : 

A user might watch multiple items, one after the other during a session. After a user watches a content, N = 2 new items are recommended. With probability q (input parameter): she ends the viewing session (i.e. does not watch another video). 
With probability 1-q, she proceeds with one more video as follows:

--> f ALL N recommended are "relevant" (i.e., have higher u_ij than u_min), then with probability α (input parameter): the user picks one of the N recommended items (with equal probability). Else,
With probability 1-α: the users picks any item k from the entire catalogue of K items with probability p_k (you can assume for simplicity that p_k = 1/K....i.e. uniform).

--> If at least one of the N recommendations is irrelevant, then again the users picks any item k from the entire catalogue of K items with probability p_k.

The objective was to minimize the average total cost of items watched during a session and at the same time keep the user happy ( suggest videos with relevance ).

We used Policy iteration algorithm and Q-learning, and compared them through their resutls.

# Phase 2

We observed in the first phase that for great K ( K > 100 ), the Q-learning breakes. So, for the second phase we had to optimize the algorithm. The goal was to use a variation of Q-learning to achieve the same optimality, but for great K.
The algorithm we implemented was the SlateQ ( developed from Google ). Keep in mind that the enviroment paramateres and the rules we used for the phase 1, remained the same.
