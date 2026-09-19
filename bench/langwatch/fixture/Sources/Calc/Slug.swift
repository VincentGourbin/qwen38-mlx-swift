import Foundation

/// URL-friendly identifiers derived from free text.
public enum Slug {
    /// Slugs longer than this are cut at a word boundary.
    public static let maxLength = 48

    /// Lowercases, strips accents, and turns runs of non-alphanumerics into a single hyphen.
    public static func slugify(_ text: String) -> String {
        let folded = text.folding(options: [.diacriticInsensitive, .caseInsensitive], locale: .current)
        var result = ""
        var pendingHyphen = false
        for scalar in folded.unicodeScalars {
            if CharacterSet.alphanumerics.contains(scalar) {
                if pendingHyphen, !result.isEmpty { result.append("-") }
                pendingHyphen = false
                result.unicodeScalars.append(scalar)
            } else {
                pendingHyphen = true
            }
        }
        return truncate(result)
    }

    private static func truncate(_ slug: String) -> String {
        guard slug.count > maxLength else { return slug }
        let cut = slug.prefix(maxLength)
        if let lastHyphen = cut.lastIndex(of: "-") { return String(cut[..<lastHyphen]) }
        return String(cut)
    }
}
