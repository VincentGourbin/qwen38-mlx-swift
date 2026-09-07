# M2 — MTP persistant et conversation multi-tour

> État : étude locale après validation M1, 2026-08-29.

## Conclusion

Le chemin M1 (`MLXLMCommon.generate(input:context:mtpDrafter:blockSize:)`)
rejoue le prompt complet et possède l'état du drafter uniquement dans
`MTPSpeculativeTokenIterator`. Il ne peut donc pas être branché directement sur
la conversation chaude de `ChatSession`.

Le premier tour image peut utiliser MTP. Pour un historique contenant une image,
le runtime doit actuellement repasser en génération standard : le drafter Qwen
est prérempli avec une position 1-D et ne reçoit pas la table complète des
positions M-RoPE 3 axes du préfixe image. Ce fallback est intentionnel et doit
rester fermé tant que M2 n'a pas les positions et les états persistants requis.

## État à transporter par une conversation M2

Une session MTP persistante doit conserver, dans un même objet sérialisé par
l'acteur du runtime :

1. le cache hybride cible (`KVCacheSimple` + `MambaCache`) et son
   `LMOutput.State` ;
2. le cache privé full-attention du drafter et `MTPDrafterState` ;
3. le ledger exact des tokens représentés par chaque cache, y compris les
   tokens de sortie encore non engagés après une vérification spéculative ;
4. la dernière représentation cachée cible utilisable comme frontière du
   prochain tour ;
5. les positions absolues/M-RoPE et leurs deltas pour le préfixe multimodal ;
6. les messages structurés et les références média nécessaires au rendu du
   prochain suffixe.

Ces éléments ne doivent jamais être sauvegardés séparément : un cache sans son
`LMOutput.State` peut reprendre un VLM à une position incohérente.

## Couture upstream nécessaire

Le protocole `StatefulMTPDrafterModel` sait déjà gérer le cycle
`prepare → draftBlock → commit`, mais son `prepareDrafterState` suppose un
préfixe froid et remet `nextPosition` à zéro. M2 doit ajouter une opération de
continuation explicite (nom à valider dans upstream), équivalente à :

```text
appendDrafterState(
    target,
    suffixTokens,
    suffixTargetHidden,
    boundaryTargetHidden,
    positionDeltas,
    state,
    sampler
)
```

Cette opération doit :

- ne pas recréer le cache privé du drafter ;
- ne pas remettre sa position à zéro ;
- invalider le `seedToken` lorsqu'un nouveau message utilisateur s'intercale ;
- traiter le suffixe avec les positions absolues de la cible ;
- rétablir `targetHidden` de frontière après chaque rejet partiel.

En parallèle, l'itérateur MTP doit accepter un état initial et exposer un
snapshot de fin de tour. Le snapshot doit être atomique : caches cible/drafter,
états, ledger et compteurs de tokens. Une simple copie du tableau `[KVCache]`
ne suffit pas.

## Découpage d'implémentation

### M2-a — texte sans média

- ajouter le snapshot/restauration de l'itérateur ;
- ajouter le chemin suffixe texte dans le contrôleur de conversation ;
- valider deux tours texte en greedy, avec et sans MTP, token à token ;
- valider un rejet partiel GDN et l'offset de chaque cache après reprise.

### M2-b — image au premier tour, texte au suivant

- conserver les `positionIds` et `ropeDeltas` produits par le premier préfill ;
- vérifier que le suffixe texte reprend à la position absolue correcte ;
- comparer à une référence froide qui rejoue le même historique ;
- ne réactiver MTP qu'après une parité complète.

### M2-c — nouvelle image et reset

- nouvelle image : reconstruire le suffixe multimodal avec son propre bloc
  d'embeddings, sans réinjecter l'ancienne image dans le drafter ;
- tant que cette voie n'est pas prouvée, fallback standard mesuré ;
- reset : libérer les caches et recréer les deux états, en gardant les poids
  chargés.

### M2-d — profondeur de draft

- passer de `blockSize = 2` à 3, puis 5, puis 9 ;
- tester chaque profondeur avec rejet à la première position et rejet tardif ;
- mesurer acceptation, tokens vérifiés, débit effectif, TTFT et mémoire ;
- ne publier une profondeur que si la sortie greedy reste identique.

## Critère de sortie M2

Pour chaque variante 4-bit et 8-bit :

- trois tours avec image au premier tour, puis deux tours texte ;
- sortie identique en greedy avec MTP activé/désactivé ;
- aucune réinjection média incorrecte ;
- accept rate, rounds, tokens proposés/acceptés et fallback affichés ;
- TTFT et débit issus exclusivement de `MLXProfiler` ;
- tests et binaire construits par `xcodebuild` avec `default.metallib`.

