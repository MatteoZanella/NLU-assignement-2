# NLU assignment n.2 - Report
At the beginning, SpaCy is updated to >=3.0 in order to use the new alignment functionalities. Then, the dataset and the conll.py file with all the functions is downloaded.

The dataset is converted from the txt files using `read_corpus_conll()` function. Since the provided splits are not used for training, all the examples can be used for the evaluation. After loading all the three files, the sentences containing only metadata (the ones starting with `-DOCSTART-`) were removed. The entire dataset is randomly sampled down to 8000 sentences in order to be faster in the demonstration. For a complete analysis, the full dataset could be used.


## Task 1: SpaCy NER Evaluation

### Part 1.1: SpaCy NER and alignment
In order to have all the needed informations inside the SpaCy Docs, in the first part I set a string extension to the Tokens with `Token.set_extension("ent_ref", default='')`, so the reference entity tags can be direcly saved in the attribute `token._.ent_ref`.

Then I defined the function `to_ref_entity(token)` that given a SpaCy Token, returns its IOB tag in the format (I|O|B)-(LOC|ORG|PER|MISC). Since the SpaCy classification is more fine-grained, I had to group labels together:
- `GPE|FAC|LOC` -> `LOC`: Includes all SpaCy kinds of locations
- `ORG` -> `ORG`: The label is the same
- `PERSON` -> `PER`: The label is less verbose
- `...` -> `MISC`: All the other labels are converted into the miscellaneous category

Then, the entire dataset is evaluated with SpaCy. The successive problem is the tokenization alignment needed to verify that the tokens are assigned to the same NE: I verified that SpaCy tokenizer always operates a more aggressive split than the one considered by the ground truth. Fortunately, the new version of SpaCy provides the Alignment object that can be used to obtain the tokens that should be split and merged to convert one tokenization into another. Since the SpaCy tokenizer is more aggressive, only a merge operation is needed. The `doc.retokenize()` exposes a `merge()` operation that automatically handles the re-assignment of the NE from the old tokens to the merged one. After the alignement process, the reference entities are saved token by token using the custom `._.ent_ref` property.

### Part 1.2: Token-level performance
For this part, I simpy produced two lists with the spacy translated predictions and the reference ones, as required by `sklearn.classification_report()`. Additionaly, I also provided a confusion matrix.

### Part 1.3: Chunk-level performance
For the chunk-level performance, I produced two lists holding the tuples `(text, NE)` as required by the conll.py function `evaluate()`.
The worst performance is for the `MISC` tag, which is probably caused by the diversity in granularity.


## Task 2: Grouping of Entities
The `grouped_entities(sentence)` function initially takes the sentence string and converts it into a SpaCy Doc. Then, the noun chunks list is extracted with the SpaCy function.
The function keeps track the first of these chunks which has not been fully explored yet, and exactly which of its tokens should be explored next.
Then, for each token in the doc, the function checks if it token within a noun chunk that should be explored next. If not, its type (if exists) is appended as a single item list. Otherwise, it's added to a set holding the current noun chunk entities. When the last token of the current noun chunk is reached, the set is converted to an ordered list and added to the entities; since the objective is only recognising which types appear together, ordering the list permits to unify cases in which they appear together in different order.

### Part 2.1: Frequency analysis
This part makes use of the python Counter collection to count every time a certain combination of types appears. The keys of the Counter dictionary is a string composed by the types of the inner list separated by a dash. The results are then printed from the most common to the least.


## Task 3: Covering full noun-compounds
The aim of the `expand_entities(doc)` is expanding the entity spans to cover the full noun compunds reachable with the 'compound' dependency relationship.
The function processes entity by entity in the `doc.ents` iterator, trying to expand it. 
The (possibly) expanded entity is then saved in a list with Span(doc, entity_start_idx, entity_end_idx, label=ent.label_) and the list of all entities is finally applied to the Doc with the method `doc.set_ents(entities)`.
The expansion follows the following rules: 
- The expanded entity can include at most all the dependency descendants of the starting entity tokens, excluding the tokens that are part of another entity. This is important because otherwise entities could overlap.
- A candidate descendant token is included in the entity if there is a sequence of `compound` relationships that connect it to a starting entity token.
- If a candidate token is selected, all the tokens between it and the original entity tokens are selected as well.

The first rule is enforced by taking the extremes of the subtree of the root of the entity span, which are cropped if intersecting with the previous or the next entity boundaries. 
Then, from the starting extreme `search_start` towards the first token of the entity, it is checked if these candidate tokens respect the second rule. If not, the neighbor token is selected, otherwise its index is selected as the starting point for the expanded entity (including all the enclosed tokens, according to the third rule).
The search from the `search_end` extreme behave similarly, but it starts from the further right token and proceeds backward towards the final token of the starting entity.
At this point, `search_start` and `search_end` indices encompass the new entity span.

This postprocessing can be directly applied to the `docs` variable used for Task 1, so it's already correctly tokenized for the evaluation.

### Part 2.1: Evaluation (optional)
The evaluation is performed similarly to the one in Part 1.2 and Part 1.3. The results are slightly worse that the previous results. For instance, in one sentence the NE "Shimon Peres" is extended to "minister Shimon Peres" since minister has a compound relationship with Shimon, and that is clearly not part of the PERSON type.
