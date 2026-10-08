// GT Display Script
// Reads JSON from element with id="GT" and displays in a readable overlay

(function() {
    // Read JSON from GT element
    const gtElement = document.getElementById('GT');
    let gtData = null;

    if (gtElement) {
        try {
            gtData = JSON.parse(gtElement.textContent);
        } catch (e) {
            console.error('Failed to parse JSON from GT element:', e);
        }
    }

    // Generate distinct colors for groups
    function generateColor(index) {
        const hue = (index * 137.508) % 360; // Golden angle for good distribution
        return `hsla(${hue}, 70%, 60%, 0.3)`;
    }

    // Generate label color (darker, more saturated)
    function generateLabelColor(index) {
        const hue = (index * 137.508) % 360;
        return `hsla(${hue}, 80%, 40%, 1)`;
    }

    // Highlight and collect MENU_<int>, PAIR_<int>, and GENERIC groups
    function highlightAndCollectGroups() {
        const groups = {}; // Structure: { "MENU_1": { elements: [...], subFields: {...} }, ... }
        const allElements = document.querySelectorAll('*');
        const groupLabels = []; // Store all label elements for toggling

        // Regex patterns for group identifiers
        const menuPattern = /^MENU_(\d+)$/;
        const pairPattern = /^PAIR_(\d+)$/;
        const genericPattern = /^GENERIC$/;

        // First pass: collect all elements and their group memberships
        allElements.forEach(element => {
            const classList = Array.from(element.classList);

            // Check if element has GENERIC class
            const hasGeneric = classList.includes('GENERIC');

            // Check for MENU_X or PAIR_X classes
            const menuClass = classList.find(cls => cls.match(menuPattern));
            const pairClass = classList.find(cls => cls.match(pairPattern));

            // Process MENU groups
            if (menuClass) {
                const groupId = menuClass;
                if (!groups[groupId]) {
                    groups[groupId] = {
                        elements: [],
                        subFields: {},
                        elementSubFields: new Map()
                    };
                }
                groups[groupId].elements.push(element);

                const elementSubFields = [];
                classList.forEach(cls => {
                    if (cls.startsWith('MENU_') && !cls.match(menuPattern)) {
                        if (!groups[groupId].subFields[cls]) {
                            groups[groupId].subFields[cls] = [];
                        }
                        groups[groupId].subFields[cls].push(element);
                        elementSubFields.push(cls);
                    }
                });
                groups[groupId].elementSubFields.set(element, elementSubFields);
            }

            // Process PAIR groups
            if (pairClass) {
                const groupId = pairClass;
                if (!groups[groupId]) {
                    groups[groupId] = {
                        elements: [],
                        subFields: {},
                        elementSubFields: new Map()
                    };
                }
                groups[groupId].elements.push(element);

                const elementSubFields = [];
                classList.forEach(cls => {
                    if (cls.startsWith('PAIR_') && !cls.match(pairPattern)) {
                        if (!groups[groupId].subFields[cls]) {
                            groups[groupId].subFields[cls] = [];
                        }
                        groups[groupId].subFields[cls].push(element);
                        elementSubFields.push(cls);
                    }
                });
                groups[groupId].elementSubFields.set(element, elementSubFields);
            }

            // Process GENERIC group
            if (hasGeneric) {
                const groupId = 'GENERIC';
                if (!groups[groupId]) {
                    groups[groupId] = {
                        elements: [],
                        subFields: {},
                        elementSubFields: new Map()
                    };
                }
                groups[groupId].elements.push(element);

                const elementSubFields = [];
                classList.forEach(cls => {
                    // For GENERIC, collect all classes that look like subfields:
                    // - Start with GENERIC_ or GEN_
                    // - Are uppercase with underscores (semantic field pattern)
                    // - Exclude: GENERIC itself, LE- prefixed classes, common utility classes
                    const isSubfield = cls !== 'GENERIC' &&
                                      !cls.startsWith('LE-') &&
                                      !cls.startsWith('layout-') &&
                                      !cls.startsWith('text-') &&
                                      !cls.startsWith('flex') &&
                                      !cls.startsWith('grid') &&
                                      !cls.startsWith('item-') &&
                                      (cls.startsWith('GENERIC_') ||
                                       cls.startsWith('GEN_') ||
                                       (/^[A-Z_]+$/.test(cls) && cls.includes('_'))); // Uppercase with underscores

                    if (isSubfield) {
                        if (!groups[groupId].subFields[cls]) {
                            groups[groupId].subFields[cls] = [];
                        }
                        groups[groupId].subFields[cls].push(element);
                        elementSubFields.push(cls);
                    }
                });
                groups[groupId].elementSubFields.set(element, elementSubFields);
            }
        });

        // Sort groups by name for consistent coloring
        const sortedGroupIds = Object.keys(groups).sort((a, b) => {
            // Extract type and number for proper sorting
            const aMatch = a.match(/^(MENU|PAIR)_(\d+)$/);
            const bMatch = b.match(/^(MENU|PAIR)_(\d+)$/);

            // Handle GENERIC separately
            if (a === 'GENERIC' && b !== 'GENERIC') return 1; // GENERIC goes last
            if (b === 'GENERIC' && a !== 'GENERIC') return -1;
            if (a === 'GENERIC' && b === 'GENERIC') return 0;

            if (aMatch && bMatch) {
                if (aMatch[1] !== bMatch[1]) {
                    return aMatch[1].localeCompare(bMatch[1]);
                }
                return parseInt(aMatch[2]) - parseInt(bMatch[2]);
            }
            return a.localeCompare(b);
        });

        // Second pass: apply highlighting with colors
        sortedGroupIds.forEach((groupId, index) => {
            const color = generateColor(index);
            const labelColor = generateLabelColor(index);

            groups[groupId].elements.forEach(element => {
                element.style.backgroundColor = color;
                element.style.transition = 'background-color 0.3s';
                element.style.position = 'relative';
                element.style.outline = `2px solid ${labelColor}`;
                element.style.outlineOffset = '-2px';

                // Get the subfields for this specific element
                const elementSubFields = groups[groupId].elementSubFields.get(element) || [];

                // Create label text: GROUP_ID + subfields
                let labelText = groupId;
                if (elementSubFields.length > 0) {
                    labelText += ' | ' + elementSubFields.join(', ');
                }

                // Add a label above and to the right of the element
                const label = document.createElement('div');
                label.textContent = labelText;
                label.className = 'group-label'; // Add class for easy toggling
                label.style.position = 'absolute';
                label.style.top = '-20px';
                label.style.right = '0';
                label.style.color = labelColor;
                label.style.fontWeight = 'bold';
                label.style.fontSize = '9px';
                label.style.backgroundColor = 'rgba(255, 255, 255, 0.95)';
                label.style.padding = '2px 6px';
                label.style.borderRadius = '3px';
                label.style.whiteSpace = 'nowrap';
                label.style.pointerEvents = 'none';
                label.style.zIndex = '1000';
                label.style.boxShadow = '0 2px 4px rgba(0,0,0,0.3)';
                label.style.border = `1px solid ${labelColor}`;
                label.style.display = 'block'; // Initially visible

                element.appendChild(label);
                groupLabels.push(label);
            });
        });

        return { groups, sortedGroupIds, groupLabels };
    }

    // Display group information in overlay
    function displayGroupInfo(groups, sortedGroupIds, container) {
        if (sortedGroupIds.length === 0) {
            return;
        }

        const groupSection = document.createElement('div');
        groupSection.style.marginTop = '15px';
        groupSection.style.paddingTop = '12px';
        groupSection.style.borderTop = '2px solid rgba(255, 255, 255, 0.4)';

        const groupTitle = document.createElement('div');
        groupTitle.textContent = `Element Groups (${sortedGroupIds.length})`;
        groupTitle.style.fontWeight = 'bold';
        groupTitle.style.fontSize = '12px';
        groupTitle.style.marginBottom = '10px';
        groupTitle.style.color = 'rgba(255, 200, 100, 1)';
        groupSection.appendChild(groupTitle);

        sortedGroupIds.forEach((groupId, index) => {
            const group = groups[groupId];
            const color = generateColor(index);
            const labelColor = generateLabelColor(index);

            const groupContainer = document.createElement('div');
            groupContainer.style.marginBottom = '12px';
            groupContainer.style.paddingBottom = '8px';
            groupContainer.style.borderBottom = '1px solid rgba(255, 255, 255, 0.2)';

            // Group header with color indicator
            const groupHeader = document.createElement('div');
            groupHeader.style.display = 'flex';
            groupHeader.style.alignItems = 'center';
            groupHeader.style.marginBottom = '6px';

            const colorBox = document.createElement('div');
            colorBox.style.width = '16px';
            colorBox.style.height = '16px';
            colorBox.style.backgroundColor = color;
            colorBox.style.border = `2px solid ${labelColor}`;
            colorBox.style.borderRadius = '3px';
            colorBox.style.marginRight = '8px';
            colorBox.style.flexShrink = '0';

            const groupLabel = document.createElement('span');
            groupLabel.textContent = `${groupId} (${group.elements.length} element${group.elements.length !== 1 ? 's' : ''})`;
            groupLabel.style.fontWeight = 'bold';
            groupLabel.style.fontSize = '11px';
            groupLabel.style.color = labelColor;

            groupHeader.appendChild(colorBox);
            groupHeader.appendChild(groupLabel);
            groupContainer.appendChild(groupHeader);

            // Sub-fields
            if (Object.keys(group.subFields).length > 0) {
                const subFieldsContainer = document.createElement('div');
                subFieldsContainer.style.marginLeft = '24px';
                subFieldsContainer.style.fontSize = '10px';

                const subFieldsList = document.createElement('div');
                subFieldsList.textContent = 'Sub-fields: ' + Object.keys(group.subFields).sort().join(', ');
                subFieldsList.style.color = 'rgba(200, 200, 200, 1)';
                subFieldsList.style.marginBottom = '4px';
                subFieldsContainer.appendChild(subFieldsList);

                // Show content from sub-fields
                Object.entries(group.subFields).sort().forEach(([subField, elements]) => {
                    const subFieldRow = document.createElement('div');
                    subFieldRow.style.marginTop = '3px';

                    const subFieldName = document.createElement('span');
                    subFieldName.textContent = subField + ': ';
                    subFieldName.style.color = 'rgba(150, 200, 255, 0.9)';
                    subFieldName.style.fontWeight = 'normal';

                    const subFieldValues = document.createElement('span');
                    const values = elements
                        .map(el => el.textContent.trim())
                        .filter(text => text.length > 0)
                        .slice(0, 3); // Limit to first 3 values

                    if (values.length > 0) {
                        subFieldValues.textContent = values.join(', ') + (elements.length > 3 ? '...' : '');
                        subFieldValues.style.color = 'rgba(100, 255, 150, 1)';
                    } else {
                        subFieldValues.textContent = '(empty)';
                        subFieldValues.style.color = 'rgba(150, 150, 150, 0.8)';
                    }

                    subFieldRow.appendChild(subFieldName);
                    subFieldRow.appendChild(subFieldValues);
                    subFieldsContainer.appendChild(subFieldRow);
                });

                groupContainer.appendChild(subFieldsContainer);
            }

            groupSection.appendChild(groupContainer);
        });

        container.appendChild(groupSection);
    }

    // Highlight elements containing exact value matches
    function highlightValues(values) {
        // Get all text-containing elements (excluding script tags and the overlay)
        const allElements = document.body.querySelectorAll('*:not(script):not(style)');

        allElements.forEach(element => {
            // Get the direct text content (not including children)
            const textContent = Array.from(element.childNodes)
                .filter(node => node.nodeType === Node.TEXT_NODE)
                .map(node => node.textContent.trim())
                .join(' ');

            // Check if this element's text exactly matches any of the values
            for (const value of values) {
                if (textContent === value || element.textContent.trim() === value) {
                    element.style.backgroundColor = 'rgba(0, 100, 255, 0.3)';
                    element.style.transition = 'background-color 0.3s';
                    break;
                }
            }
        });
    }

    // Display structured format (header/question/answer)
    function displayStructuredFormat(data, container) {
        for (const [pairKey, pairData] of Object.entries(data)) {
            const pairContainer = document.createElement('div');
            pairContainer.style.marginBottom = '12px';
            pairContainer.style.paddingBottom = '8px';
            pairContainer.style.borderBottom = '1px solid rgba(255, 255, 255, 0.2)';

            // Pair identifier (e.g., PAIR_1)
            const pairLabel = document.createElement('div');
            pairLabel.textContent = pairKey;
            pairLabel.style.fontSize = '10px';
            pairLabel.style.color = 'rgba(255, 255, 255, 0.6)';
            pairLabel.style.marginBottom = '4px';
            pairContainer.appendChild(pairLabel);

            // Header
            if (pairData.header) {
                const header = document.createElement('div');
                header.innerHTML = `<strong>Header:</strong> ${pairData.header}`;
                header.style.marginBottom = '3px';
                pairContainer.appendChild(header);
            }

            // Question
            if (pairData.question) {
                const question = document.createElement('div');
                question.innerHTML = `<strong>Question:</strong> ${pairData.question}`;
                question.style.marginBottom = '3px';
                pairContainer.appendChild(question);
            }

            // Answer
            if (pairData.answer) {
                const answer = document.createElement('div');
                answer.innerHTML = `<strong>Answer:</strong> ${pairData.answer}`;
                answer.style.color = 'rgba(100, 255, 150, 1)';
                pairContainer.appendChild(answer);
            }

            container.appendChild(pairContainer);
        }
    }

    // Display simple key-value format
    function displaySimpleFormat(data, container) {
        const table = document.createElement('div');

        for (const [key, value] of Object.entries(data)) {
            const row = document.createElement('div');
            row.style.display = 'flex';
            row.style.marginBottom = '6px';
            row.style.paddingBottom = '6px';
            row.style.borderBottom = '1px dotted rgba(255, 255, 255, 0.2)';

            const keySpan = document.createElement('span');
            keySpan.textContent = key + ':';
            keySpan.style.fontWeight = 'bold';
            keySpan.style.minWidth = '100px';
            keySpan.style.color = 'rgba(150, 200, 255, 1)';

            const valueSpan = document.createElement('span');
            valueSpan.textContent = value;
            valueSpan.style.marginLeft = '10px';
            valueSpan.style.color = 'rgba(100, 255, 150, 1)';

            row.appendChild(keySpan);
            row.appendChild(valueSpan);
            table.appendChild(row);
        }

        container.appendChild(table);
    }

    // Display nested objects format (e.g., MENU_1, MENU_2, etc.)
    function displayNestedFormat(data, container) {
        for (const [groupKey, groupData] of Object.entries(data)) {
            const groupContainer = document.createElement('div');
            groupContainer.style.marginBottom = '14px';
            groupContainer.style.paddingBottom = '10px';
            groupContainer.style.borderBottom = '1px solid rgba(255, 255, 255, 0.3)';

            // Group identifier (e.g., MENU_1, VOID_MENU, GENERIC)
            const groupLabel = document.createElement('div');
            groupLabel.textContent = groupKey;
            groupLabel.style.fontSize = '11px';
            groupLabel.style.fontWeight = 'bold';
            groupLabel.style.color = 'rgba(255, 200, 100, 1)';
            groupLabel.style.marginBottom = '6px';
            groupLabel.style.paddingBottom = '4px';
            groupLabel.style.borderBottom = '1px dotted rgba(255, 255, 255, 0.2)';
            groupContainer.appendChild(groupLabel);

            // Check if groupData is an object
            if (typeof groupData === 'object' && groupData !== null && !Array.isArray(groupData)) {
                // Display nested key-value pairs
                const nestedTable = document.createElement('div');
                nestedTable.style.marginLeft = '10px';

                for (const [key, value] of Object.entries(groupData)) {
                    const row = document.createElement('div');
                    row.style.display = 'flex';
                    row.style.marginBottom = '4px';
                    row.style.fontSize = '10px';

                    const keySpan = document.createElement('span');
                    keySpan.textContent = key + ':';
                    keySpan.style.fontWeight = 'normal';
                    keySpan.style.minWidth = '150px';
                    keySpan.style.color = 'rgba(150, 200, 255, 0.9)';

                    const valueSpan = document.createElement('span');
                    valueSpan.textContent = typeof value === 'object' ? JSON.stringify(value) : value;
                    valueSpan.style.marginLeft = '10px';
                    valueSpan.style.color = 'rgba(100, 255, 150, 1)';

                    row.appendChild(keySpan);
                    row.appendChild(valueSpan);
                    nestedTable.appendChild(row);
                }

                groupContainer.appendChild(nestedTable);
            } else {
                // Handle non-object values
                const valueDiv = document.createElement('div');
                valueDiv.textContent = typeof groupData === 'object' ? JSON.stringify(groupData) : groupData;
                valueDiv.style.marginLeft = '10px';
                valueDiv.style.color = 'rgba(100, 255, 150, 1)';
                valueDiv.style.fontSize = '10px';
                groupContainer.appendChild(valueDiv);
            }

            container.appendChild(groupContainer);
        }
    }

    // Collect all values recursively from nested objects
    function collectValues(obj, values = []) {
        if (typeof obj !== 'object' || obj === null) {
            values.push(String(obj));
            return values;
        }

        for (const value of Object.values(obj)) {
            if (typeof value === 'object' && value !== null) {
                collectValues(value, values);
            } else {
                values.push(String(value));
            }
        }

        return values;
    }

    // Create overlay container
    const overlay = document.createElement('div');
    overlay.style.position = 'relative';
    overlay.style.backgroundColor = 'rgba(0, 0, 0, 0.85)';
    overlay.style.color = 'white';
    overlay.style.padding = '15px';
    overlay.style.fontSize = '11px';
    overlay.style.fontFamily = 'monospace';
    overlay.style.marginTop = '8mm';
    overlay.style.borderRadius = '6px';
    overlay.style.boxShadow = '0 2px 10px rgba(0, 0, 0, 0.4)';
    overlay.style.lineHeight = '1.6';
    overlay.style.maxHeight = '500px';
    overlay.style.overflowY = 'auto';

    let valuesToHighlight = [];

    // Only process GT data if it exists
    if (gtData) {
        // Create title
        const title = document.createElement('div');
        title.textContent = 'Ground Truth Data';
        title.style.fontWeight = 'bold';
        title.style.fontSize = '12px';
        title.style.marginBottom = '10px';
        title.style.borderBottom = '2px solid rgba(255, 255, 255, 0.4)';
        title.style.paddingBottom = '6px';
        overlay.appendChild(title);

        // Detect format and display accordingly
        const firstKey = Object.keys(gtData)[0];
        const firstValue = gtData[firstKey];

        // Check if it's the header/question/answer format
        if (typeof firstValue === 'object' && firstValue !== null &&
            ('header' in firstValue || 'question' in firstValue || 'answer' in firstValue)) {
            // Format 1: Structured pairs with header/question/answer
            displayStructuredFormat(gtData, overlay);
            // Collect answer values for highlighting
            for (const pairData of Object.values(gtData)) {
                if (pairData.answer) {
                    valuesToHighlight.push(pairData.answer);
                }
            }
        } else if (typeof firstValue === 'object' && firstValue !== null) {
            // Format 3: Nested objects (MENU_1, MENU_2, etc.)
            displayNestedFormat(gtData, overlay);
            // Collect all nested values for highlighting
            valuesToHighlight = collectValues(gtData);
        } else {
            // Format 2: Simple key-value pairs
            displaySimpleFormat(gtData, overlay);
            // Collect all values for highlighting
            valuesToHighlight = Object.values(gtData);
        }

        // Highlight elements containing the values
        highlightValues(valuesToHighlight);
    }

    // Highlight placeholder elements with red background
    highlightPlaceholders();

    // Highlight layout elements and display count
    const layoutElementCount = highlightLayoutElements();

    // Highlight and collect MENU_/PAIR_/GENERIC groups
    const { groups, sortedGroupIds, groupLabels } = highlightAndCollectGroups();

    // Add toggle button for group labels
    const toggleButton = document.createElement('button');
    toggleButton.textContent = 'Hide Group Labels';
    toggleButton.style.position = 'absolute';
    toggleButton.style.top = '10px';
    toggleButton.style.right = '10px';
    toggleButton.style.padding = '6px 12px';
    toggleButton.style.fontSize = '10px';
    toggleButton.style.fontWeight = 'bold';
    toggleButton.style.backgroundColor = 'rgba(100, 150, 255, 0.9)';
    toggleButton.style.color = 'white';
    toggleButton.style.border = 'none';
    toggleButton.style.borderRadius = '4px';
    toggleButton.style.cursor = 'pointer';
    toggleButton.style.boxShadow = '0 2px 4px rgba(0,0,0,0.3)';
    toggleButton.style.pointerEvents = 'auto';
    toggleButton.style.zIndex = '1001';

    let labelsVisible = true;
    toggleButton.addEventListener('click', function() {
        labelsVisible = !labelsVisible;
        groupLabels.forEach(label => {
            label.style.display = labelsVisible ? 'block' : 'none';
        });
        toggleButton.textContent = labelsVisible ? 'Hide Group Labels' : 'Show Group Labels';
    });

    overlay.appendChild(toggleButton);

    // Add layout element count to overlay
    const layoutCountDiv = document.createElement('div');
    layoutCountDiv.textContent = `Layout Elements: ${layoutElementCount}`;
    layoutCountDiv.style.fontSize = '11px';
    layoutCountDiv.style.fontWeight = 'bold';
    layoutCountDiv.style.color = 'rgba(255, 255, 100, 1)';
    layoutCountDiv.style.marginTop = '10px';
    layoutCountDiv.style.paddingTop = '8px';
    layoutCountDiv.style.borderTop = '1px solid rgba(255, 255, 255, 0.3)';
    overlay.appendChild(layoutCountDiv);

    // Display group information in overlay
    displayGroupInfo(groups, sortedGroupIds, overlay);

    document.body.appendChild(overlay);

    // Highlight elements with data-placeholder attribute
    function highlightPlaceholders() {
        const placeholderElements = document.querySelectorAll('[data-placeholder]');
        placeholderElements.forEach(element => {
            element.style.backgroundColor = 'rgba(255, 0, 0, 0.3)';
            element.style.transition = 'background-color 0.3s';
            element.style.position = 'relative';

            // Get the placeholder value
            const placeholderValue = element.getAttribute('data-placeholder');

            // Create a label to display the placeholder value
            const label = document.createElement('div');
            label.textContent = placeholderValue;
            label.style.position = 'absolute';
            label.style.top = '50%';
            label.style.left = '50%';
            label.style.transform = 'translate(-50%, -50%)';
            label.style.color = 'red';
            label.style.fontWeight = 'bold';
            label.style.fontSize = '10px';
            label.style.backgroundColor = 'rgba(255, 255, 255, 0.8)';
            label.style.padding = '2px 6px';
            label.style.borderRadius = '3px';
            label.style.whiteSpace = 'nowrap';
            label.style.pointerEvents = 'none';
            label.style.zIndex = '1000';

            element.appendChild(label);
        });
    }

    // Highlight layout-element elements and return count
    function highlightLayoutElements() {
        // Find all elements with classes starting with LE-
        const allElements = document.querySelectorAll('*');
        const layoutElements = Array.from(allElements).filter(element => {
            return Array.from(element.classList).some(cls => cls.startsWith('LE-'));
        });

        layoutElements.forEach(element => {
            // Highlight with transparent yellow background
            element.style.backgroundColor = 'rgba(255, 255, 0, 0.3)';
            element.style.transition = 'background-color 0.3s';
            element.style.position = 'relative';

            // Extract the type from the classList (e.g., LE-TEXT, LE-TABLE, LE-TITLE)
            const classList = Array.from(element.classList);
            const typeClass = classList.find(cls => cls.startsWith('LE-'));

            if (typeClass) {
                // Create a label to display the type
                const label = document.createElement('div');
                label.textContent = typeClass.toUpperCase();
                label.style.position = 'absolute';
                label.style.top = '5px';
                label.style.left = '5px';
                label.style.color = 'rgba(200, 200, 0, 1)';
                label.style.fontWeight = 'bold';
                label.style.fontSize = '10px';
                label.style.backgroundColor = 'rgba(0, 0, 0, 0.5)';
                label.style.padding = '2px 6px';
                label.style.borderRadius = '3px';
                label.style.whiteSpace = 'nowrap';
                label.style.pointerEvents = 'none';
                label.style.zIndex = '1000';

                element.appendChild(label);
            }
        });

        return layoutElements.length;
    }
})();