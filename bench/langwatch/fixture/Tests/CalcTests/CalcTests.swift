import XCTest
@testable import Calc

final class CalcTests: XCTestCase {
    func testAverage() {
        XCTAssertEqual(Calc.average([2, 4, 6]), 4)
    }

    func testAverageOfEmptyIsZero() {
        // The empty list must not divide by zero: the expected value is 0.
        XCTAssertEqual(Calc.average([]), 0)
    }

    func testClamp() {
        XCTAssertEqual(Calc.clamp(15, to: 0...10), 10)
        XCTAssertEqual(Calc.clamp(-3, to: 0...10), 0)
    }

    func testTriangular() {
        XCTAssertEqual(Calc.triangular(4), 10)
        XCTAssertEqual(Calc.triangular(0), 0)
    }

    func testSlugify() {
        XCTAssertEqual(Slug.slugify("Été à Paris !"), "ete-a-paris")
        XCTAssertEqual(Slug.slugify("  hello   world "), "hello-world")
    }
}
