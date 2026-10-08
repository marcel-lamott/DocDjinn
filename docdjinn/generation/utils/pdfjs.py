MEASURE_DIMENSIONS_V1 = """
                            () => {
                                const body = document.body;
                                const html = document.documentElement;

                                // Force layout calculation
                                body.offsetHeight;

                                // Get body's computed style to extract margins
                                const bodyStyle = window.getComputedStyle(body);
                                const marginTop = parseFloat(bodyStyle.marginTop) || 0;
                                const marginBottom = parseFloat(bodyStyle.marginBottom) || 0;
                                const marginLeft = parseFloat(bodyStyle.marginLeft) || 0;
                                const marginRight = parseFloat(bodyStyle.marginRight) || 0;

                                const bodyRect = body.getBoundingClientRect();

                                // Find the furthest extent of content
                                let maxY = bodyRect.bottom;
                                let maxX = bodyRect.right;

                                const allElements = body.querySelectorAll('*');
                                allElements.forEach(el => {
                                    const rect = el.getBoundingClientRect();
                                    const style = window.getComputedStyle(el);

                                    if (style.display === 'none' ||
                                        style.visibility === 'hidden' ||
                                        rect.width === 0 || rect.height === 0) {
                                        return;
                                    }

                                    if (rect.bottom > maxY) maxY = rect.bottom;
                                    if (rect.right > maxX) maxX = rect.right;
                                });

                                // CRITICAL FIX: Include body margins in the total size
                                // The PDF needs to be tall enough to contain the margins too!
                                const totalWidth = Math.ceil(maxX - bodyRect.left + marginRight + 5);
                                const totalHeight = Math.ceil(maxY + marginBottom + 5);

                                return {
                                    width: totalWidth,
                                    height: totalHeight,
                                    debug: {
                                        marginTop,
                                        marginBottom,
                                        marginLeft,
                                        marginRight,
                                        bodyRectTop: bodyRect.top,
                                        bodyRectBottom: bodyRect.bottom,
                                        maxY,
                                        contentHeightWithoutMargin: Math.ceil(maxY - bodyRect.top)
                                    }
                                };
                            }
                            """

MEASURE_DIMENSIONS_V2 = """
() => {
    const body = document.body;
    const html = document.documentElement;

    // Force layout calculation
    body.offsetHeight;
    html.offsetHeight;

    // Get body's computed style to extract margins
    const bodyStyle = window.getComputedStyle(body);
    const marginTop = parseFloat(bodyStyle.marginTop) || 0;
    const marginBottom = parseFloat(bodyStyle.marginBottom) || 0;
    const marginLeft = parseFloat(bodyStyle.marginLeft) || 0;
    const marginRight = parseFloat(bodyStyle.marginRight) || 0;

    // Strategy: Find the bounding box of ALL visible content
    // This works for narrow receipts, wide tables, multi-column, everything

    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;

    // Check body itself
    const bodyRect = body.getBoundingClientRect();
    if (bodyRect.width > 0 && bodyRect.height > 0) {
        minX = Math.min(minX, bodyRect.left);
        minY = Math.min(minY, bodyRect.top);
        maxX = Math.max(maxX, bodyRect.right);
        maxY = Math.max(maxY, bodyRect.bottom);
    }

    // Check all elements to find true content bounds
    const allElements = document.querySelectorAll('*');
    allElements.forEach(el => {
        const rect = el.getBoundingClientRect();
        const style = window.getComputedStyle(el);

        // Skip hidden elements
        if (style.display === 'none' ||
            style.visibility === 'hidden' ||
            rect.width === 0 ||
            rect.height === 0) {
            return;
        }

        // Skip script/style tags
        if (el.tagName === 'SCRIPT' || el.tagName === 'STYLE') {
            return;
        }

        minX = Math.min(minX, rect.left);
        minY = Math.min(minY, rect.top);
        maxX = Math.max(maxX, rect.right);
        maxY = Math.max(maxY, rect.bottom);
    });

    // Fallback if no content found
    if (minX === Infinity) {
        minX = 0;
        minY = 0;
        maxX = bodyRect.right;
        maxY = bodyRect.bottom;
    }

    // Calculate total dimensions
    // Width: from leftmost to rightmost content + right margin
    // Height: from topmost to bottommost content + bottom margin
    const buffer = 5; // Small safety buffer

    const totalWidth = Math.ceil(maxX - minX + marginRight + buffer);
    const totalHeight = Math.ceil(maxY - minY + marginBottom + buffer);

    return {
        width: totalWidth,
        height: totalHeight,
        debug: {
            marginTop,
            marginBottom,
            marginLeft,
            marginRight,
            minX,
            minY,
            maxX,
            maxY,
            bodyWidth: bodyRect.width,
            bodyHeight: bodyRect.height
        }
    };
}
"""

MEASURE_DIMENSIONS_V3 = """
() => {
    const body = document.body;

    // Force layout
    body.offsetHeight;

    const bodyRect = body.getBoundingClientRect();

    // For receipts/documents with body padding, the body rect already includes everything
    // Just add a small buffer
    const buffer = 5;

    return {
        width: Math.ceil(bodyRect.width + buffer),
        height: Math.ceil(bodyRect.height + buffer)
    };
}
"""


MEASURE_DIMENSIONS_V4 = """
() => {
    const body = document.body;
    const html = document.documentElement;

    // Force layout calculation
    body.offsetHeight;
    html.offsetHeight;

    // Get body's computed style to extract margins
    const bodyStyle = window.getComputedStyle(body);
    const marginTop = parseFloat(bodyStyle.marginTop) || 0;
    const marginBottom = parseFloat(bodyStyle.marginBottom) || 0;
    const marginLeft = parseFloat(bodyStyle.marginLeft) || 0;
    const marginRight = parseFloat(bodyStyle.marginRight) || 0;

    // Get body padding as well
    const paddingTop = parseFloat(bodyStyle.paddingTop) || 0;
    const paddingBottom = parseFloat(bodyStyle.paddingBottom) || 0;
    const paddingLeft = parseFloat(bodyStyle.paddingLeft) || 0;
    const paddingRight = parseFloat(bodyStyle.paddingRight) || 0;

    // Strategy: Find the bounding box of ALL visible content
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;

    // Check body itself
    const bodyRect = body.getBoundingClientRect();
    if (bodyRect.width > 0 && bodyRect.height > 0) {
        minX = Math.min(minX, bodyRect.left);
        minY = Math.min(minY, bodyRect.top);
        maxX = Math.max(maxX, bodyRect.right);
        maxY = Math.max(maxY, bodyRect.bottom);
    }

    // Check all elements to find true content bounds
    const allElements = document.querySelectorAll('*');
    allElements.forEach(el => {
        const rect = el.getBoundingClientRect();
        const style = window.getComputedStyle(el);

        // Skip hidden elements
        if (style.display === 'none' ||
            style.visibility === 'hidden' ||
            rect.width === 0 ||
            rect.height === 0) {
            return;
        }

        // Skip script/style tags
        if (el.tagName === 'SCRIPT' || el.tagName === 'STYLE') {
            return;
        }

        minX = Math.min(minX, rect.left);
        minY = Math.min(minY, rect.top);
        maxX = Math.max(maxX, rect.right);
        maxY = Math.max(maxY, rect.bottom);
    });

    // Fallback if no content found
    if (minX === Infinity) {
        minX = 0;
        minY = 0;
        maxX = bodyRect.right;
        maxY = bodyRect.bottom;
    }

    // Calculate total dimensions
    // CRITICAL FIX: The viewport starts at 0,0 but content might be offset
    // We need the full document size, not just content span

    // For width: take the maximum of either the rightmost content or body width
    // For height: take the maximum of either the bottommost content or body height
    const buffer = 5;

    // Option A: Measure from viewport origin (0,0) to furthest content
    const totalWidth = Math.ceil(maxX + buffer);
    const totalHeight = Math.ceil(maxY + buffer);

    // Option B: Also consider body's full width (in case body is wider than content)
    const bodyFullWidth = bodyRect.width;
    const bodyFullHeight = bodyRect.height;

    // Use whichever is larger
    const finalWidth = Math.max(totalWidth, bodyFullWidth);
    const finalHeight = Math.max(totalHeight, bodyFullHeight);

    return {
        width: finalWidth,
        height: finalHeight,
        debug: {
            marginTop,
            marginBottom,
            marginLeft,
            marginRight,
            paddingTop,
            paddingBottom,
            paddingLeft,
            paddingRight,
            minX,
            minY,
            maxX,
            maxY,
            bodyWidth: bodyRect.width,
            bodyHeight: bodyRect.height,
            bodyLeft: bodyRect.left,
            bodyTop: bodyRect.top,
            totalWidth,
            totalHeight,
            bodyFullWidth,
            bodyFullHeight
        }
    };
}
"""

MEASURE_DIMENSIONS = """
() => {
    const body = document.body;
    const html = document.documentElement;

    // Force layout
    body.offsetHeight;
    html.offsetHeight;

    const bodyStyle = window.getComputedStyle(body);
    const paddingTop = parseFloat(bodyStyle.paddingTop) || 0;
    const paddingBottom = parseFloat(bodyStyle.paddingBottom) || 0;
    const paddingLeft = parseFloat(bodyStyle.paddingLeft) || 0;
    const paddingRight = parseFloat(bodyStyle.paddingRight) || 0;

    // Strategy: Find bounding box of ALL visible content
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;

    const bodyRect = body.getBoundingClientRect();

    // Check all elements (not just body children, in case of deep nesting)
    const allElements = document.querySelectorAll('body *');
    let hasContent = false;

    allElements.forEach(el => {
        // Skip scripts, styles, and hidden elements
        if (el.tagName === 'SCRIPT' || el.tagName === 'STYLE') return;

        const rect = el.getBoundingClientRect();
        const style = window.getComputedStyle(el);

        if (style.display === 'none' || style.visibility === 'hidden' ||
            rect.width === 0 || rect.height === 0) {
            return;
        }

        hasContent = true;
        minX = Math.min(minX, rect.left);
        minY = Math.min(minY, rect.top);
        maxX = Math.max(maxX, rect.right);
        maxY = Math.max(maxY, rect.bottom);
    });

    // Fallback if no content found
    if (!hasContent || minX === Infinity) {
        return {
            width: Math.ceil(bodyRect.width + 5),
            height: Math.ceil(bodyRect.height + 5)
        };
    }

    // Now decide: do we measure from content bounds or from body bounds?

    // Approach 1: Content-based (for narrow receipts)
    // Width = actual content span + left padding + right padding
    const contentWidth = maxX - minX;
    const contentHeight = maxY - minY;
    const contentBasedWidth = contentWidth + paddingLeft + paddingRight;
    const contentBasedHeight = contentHeight + paddingTop + paddingBottom;

    // Approach 2: Body-based (for full-width documents)
    // Width = body's full width
    const bodyBasedWidth = bodyRect.width;
    const bodyBasedHeight = bodyRect.height;

    // Decision logic:
    // If content is significantly narrower than body (e.g., < 70% of body width),
    // it's likely a centered narrow layout like a receipt
    // Otherwise, it's a full-width document

    const contentWidthRatio = contentWidth / bodyRect.width;
    const isNarrowCentered = contentWidthRatio < 0.7;

    let finalWidth, finalHeight;

    if (isNarrowCentered) {
        // Use content-based measurement (receipt-style)
        finalWidth = contentBasedWidth;
        finalHeight = Math.max(contentBasedHeight, bodyBasedHeight); // Use max for height
    } else {
        // Use body-based measurement (full-width document)
        finalWidth = bodyBasedWidth;
        finalHeight = bodyBasedHeight;
    }

    const buffer = 5;

    return {
        width: Math.ceil(finalWidth + buffer),
        height: Math.ceil(finalHeight + buffer),
        debug: {
            isNarrowCentered,
            contentWidthRatio: contentWidthRatio.toFixed(2),
            contentWidth,
            contentHeight,
            bodyWidth: bodyRect.width,
            bodyHeight: bodyRect.height,
            approach: isNarrowCentered ? 'content-based' : 'body-based'
        }
    };
}
"""
