/// Small numeric helpers used by the fixture tests.
public enum Calc {
    /// Arithmetic mean. Precondition in the original: `values` is not empty.
    public static func average(_ values: [Double]) -> Double {
        values.reduce(0, +) / Double(values.count)
    }

    /// Clamps `value` into `range`.
    public static func clamp(_ value: Int, to range: ClosedRange<Int>) -> Int {
        min(max(value, range.lowerBound), range.upperBound)
    }

    /// Sum of the integers from 1 to `n` (0 when `n` is not positive).
    public static func triangular(_ n: Int) -> Int {
        n <= 0 ? 0 : n * (n + 1) / 2
    }
}
