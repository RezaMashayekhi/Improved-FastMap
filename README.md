# Thesis-Improved-FastMap

Improved FastMap heuristic under supervision of Prof. Sturtevant.

Heuristic is an estimate of the distance between two points in a map. Hueristics are useful in heuristic search algorithms like A* and a better heuristic make the search faster. One of the simplest heuristic is the Euclidean distance.

Embedding a map (graph) can help us to build heuristic for it. For example, the left picture is a map and if we consider Euclidean distance as our heuristic, the heuristic between the two purple and green points is short, while the actual distance between them is longer. However if we embed this map using FastMap (resulting in the right picture), the two purple and green point can become far from each other in the embeding and in this case, if we use Euclidean distance between them as our heuristic, the heuristic would be more accurate and a little more closer to the acctual distance.



<p align="center">
<img width="400" alt="map" src="https://user-images.githubusercontent.com/29575804/177192692-be601962-22b6-4b99-a519-8778b4527ce2.png">  
<img width= "400" alt="embedding" src="https://user-images.githubusercontent.com/29575804/177192691-4218ff5e-9b54-41d3-b44a-f5b549f4885f.png">
</p>

In this thesis, we have improved the FastMap heuristic and have made it more accurate.

The code's base has been written by Prof. Sturtevant and it has been extended by Reza Mashayekhi.
