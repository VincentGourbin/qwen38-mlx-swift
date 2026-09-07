import CoreGraphics
import Foundation
import ImageIO
import MLX

public struct Qwen4ExpProcessedImage: @unchecked Sendable {
    public let pixels: MLXArray
    public let width: Int
    public let height: Int
    public let patchGrid: (height: Int, width: Int)

    public init(pixels: MLXArray, width: Int, height: Int, patchSize: Int = 16) {
        self.pixels = pixels
        self.width = width
        self.height = height
        self.patchGrid = (height / patchSize, width / patchSize)
    }
}

/// Minimal Qwen smart-resize compatible image path for the Flash-Next probe.
/// The model expects NHWC RGB values normalized to [-1, 1].
public enum Qwen4ExpImageProcessor {
    public static func load(
        from url: URL,
        maxPixels: Int = 1_003_520,
        patchSize: Int = 16,
        mergeSize: Int = 2
    ) throws -> Qwen4ExpProcessedImage {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
        else {
            throw Qwen4ExpImageProcessorError.invalidImage(url)
        }
        let factor = patchSize * mergeSize
        let sourceWidth = image.width
        let sourceHeight = image.height
        let ratio = Double(sourceWidth) / Double(sourceHeight)
        let scale = min(1.0, sqrt(Double(maxPixels) / Double(sourceWidth * sourceHeight)))
        var width = max(factor, Int((Double(sourceWidth) * scale).rounded()))
        var height = max(factor, Int((Double(sourceHeight) * scale).rounded()))
        width = max(factor, (width / factor) * factor)
        height = max(factor, (height / factor) * factor)
        // Keep extreme aspect ratios inside the same pixel budget after the
        // factor rounding, while preserving the source orientation.
        if Double(width) / Double(height) > ratio * 1.25 {
            width = max(factor, Int(Double(height) * ratio) / factor * factor)
        } else if Double(height) / Double(width) > (1 / ratio) * 1.25 {
            height = max(factor, Int(Double(width) / ratio) / factor * factor)
        }
        guard width >= factor, height >= factor else {
            throw Qwen4ExpImageProcessorError.invalidDimensions(sourceWidth, sourceHeight)
        }

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        var bytes = [UInt8](repeating: 0, count: width * height * 4)
        guard let context = CGContext(
            data: &bytes,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else {
            throw Qwen4ExpImageProcessorError.cannotCreateContext
        }
        context.interpolationQuality = .high
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))

        let raw = MLXArray(bytes).reshaped([height, width, 4])
        let rgb = raw[.ellipsis, 0..<3].asType(.float32) / 127.5 - 1.0
        let pixels = rgb.expandedDimensions(axis: 0).asType(.bfloat16)
        return Qwen4ExpProcessedImage(
            pixels: pixels, width: width, height: height, patchSize: patchSize)
    }
}

public enum Qwen4ExpImageProcessorError: LocalizedError, Equatable {
    case invalidImage(URL)
    case invalidDimensions(Int, Int)
    case cannotCreateContext

    public var errorDescription: String? {
        switch self {
        case .invalidImage(let url): return "Image Flash-Next illisible : \(url.path)"
        case .invalidDimensions(let width, let height):
            return "Dimensions image Flash-Next invalides : \(width)x\(height)"
        case .cannotCreateContext: return "Contexte CoreGraphics impossible pour l'image Flash-Next."
        }
    }
}
