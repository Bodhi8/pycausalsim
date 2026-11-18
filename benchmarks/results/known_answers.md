# Known-answer benchmark: PyCausalSim 0.6.0 vs 0.7.0

## Named graphs (Gaussian, n = 3,000): learned graph

| Graph | Method | 0.6.0 | 0.7.0 |
|---|---|---|---|
| chain a->b->c | PC | a->b, c->b (1 wrong arrow, 1 unsupported arrow) | a-b, b-c (correct) |
| chain a->b->c | GES | a-b, b-c (correct) | a-b, b-c (correct) |
| chain a->b->c | HYBRID | a-b, c->b (1 wrong arrow) | a-b, b-c (correct) |
| fork b<-a->c | PC | b->a, c->a (2 wrong arrows) | a-b, a-c (correct) |
| fork b<-a->c | GES | a-b, a-c (correct) | a-b, a-c (correct) |
| fork b<-a->c | HYBRID | b->a, c->a (2 wrong arrows) | a-b, a-c (correct) |
| collider a->c<-b | PC | a->c, b->c (correct) | a->c, b->c (correct) |
| collider a->c<-b | GES | a-b, a-c, b-c (1 extra edge) | a->c, b->c (correct) |
| collider a->c<-b | HYBRID | a-c, b-c (undirected where the data fix a direction) | a->c, b->c (correct) |
| diamond | PC | a-b, a-c, b-d, c-d (undirected where the data fix a direction) | a-b, a-c, b->d, c->d (correct) |
| diamond | GES | a-b, a-c, b-c, b-d, c-d (1 extra edge) | a-b, a-c, b->d, c->d (correct) |
| diamond | HYBRID | a-b, a-c, c->b, b-d, c-d (1 extra edge) | a-b, a-c, b->d, c->d (correct) |

## 12 random 5-node graphs (Gaussian, n = 5,000)

| Method | Measure | 0.6.0 | 0.7.0 |
|---|---|---|---|
| PC | Equivalence class exactly right (of 12) | 1 | 10 |
| PC | Arrows pointing the wrong way | 14 | 3 |
| PC | Arrows the data cannot support | 5 | 0 |
| PC | Missing edges | 1 | 1 |
| PC | Extra edges | 1 | 1 |
| GES | Equivalence class exactly right (of 12) | 2 | 11 |
| GES | Arrows pointing the wrong way | 0 | 0 |
| GES | Arrows the data cannot support | 0 | 0 |
| GES | Missing edges | 1 | 1 |
| GES | Extra edges | 16 | 1 |
| HYBRID | Equivalence class exactly right (of 12) | 0 | 11 |
| HYBRID | Arrows pointing the wrong way | 15 | 0 |
| HYBRID | Arrows the data cannot support | 0 | 0 |
| HYBRID | Missing edges | 1 | 1 |
| HYBRID | Extra edges | 4 | 0 |

## 12 random 5-node graphs (uniform noise, n = 5,000)

| Method | Measure | 0.6.0 | 0.7.0 |
|---|---|---|---|
| LINGAM | Graph exactly right (of 12) | 3 | 12 |
| LINGAM | Arrows pointing the wrong way | 1 | 0 |
| LINGAM | Missing edges | 0 | 0 |
| LINGAM | Extra edges | 14 | 0 |
| LINGAM | Self-loops | 0 | 0 |
| LINGAM | Errors | 0 | 0 |
| NOTEARS | Graph exactly right (of 12) | 0 | 11 |
| NOTEARS | Arrows pointing the wrong way | 0 | 1 |
| NOTEARS | Missing edges | 0 | 1 |
| NOTEARS | Extra edges | 66 | 0 |
| NOTEARS | Self-loops | 60 | 0 |
| NOTEARS | Errors | 0 | 0 |

## Attribution, 40,000 simulated users, four channels with known effects

True effects on conversion: email +10 points, search +4, display 0, social -3.

| Method | Version | email | search | display | social | Credit misallocated |
|---|---|---|---|---|---|---|
| Truth | | 74% | 26% | 0% | 0% | 0 points |
| shapley | 0.6.0 | 52% | 23% | 20% | 5% | 25.3 points |
| shapley | 0.7.0 | 75% | 22% | 3% | 0% | 3.6 points |
| logistic | 0.6.0 | 57% | 21% | 3% | 19% | 21.8 points |
| logistic | 0.7.0 | 72% | 24% | 3% | 0% | 3.3 points |
| linear | 0.6.0 | 57% | 22% | 3% | 18% | 21.1 points |
| linear | 0.7.0 | 73% | 24% | 3% | 0% | 3.1 points |
| uplift | 0.6.0 | 60% | 25% | 16% | 0% | 15.5 points |
| uplift | 0.7.0 | 70% | 27% | 3% | 0% | 5.0 points |

Credit misallocated: half the summed absolute gap between reported and true shares,
in percentage points of total credit.
