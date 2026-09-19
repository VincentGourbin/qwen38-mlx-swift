# Calc

Petit paquet Swift servant de terrain de jeu au banc d'essai LangWatch.

- `Sources/Calc/Calc.swift` : moyenne, borne (`clamp`), nombre triangulaire.
- `Sources/Calc/Slug.swift` : `slugify`, identifiants pour URL (minuscules, sans accents, tirets), coupés à `maxLength` caractères.
- `Tests/CalcTests` : `swift test`.

Un test échoue volontairement : `testAverageOfEmptyIsZero` (division par zéro dans `average`).
