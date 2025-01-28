#!/usr/bin/env python3

import re
import sys

def parse_cpp_for_class_vars(filename):
    """
    Naive parser: 
      1) Looks for 'class ClassName {'
      2) Skips until 'public:' or end-of-class
      3) Collects lines that appear to declare variables in the form:
         <type> var1, var2, ... , varN;
    Returns a dict of { className: [var1, var2, ...] }.
    """
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    class_vars = {}       # { className: [var1, var2, ...] }
    current_class = None
    in_class = False
    in_public_section = False
    
    # Regex to match lines like: 'int a, b, c;' or 'double x, y;'
    # Explanation:
    #   ^\s*               Start of line, allow leading whitespace
    #   ([\w:\<>]+(?:\s+[\w:\<>\*]+)?)  => captures a naive "type" (e.g. "int", "double", "std::vector<int>", "int*") 
    #   \s+                Some whitespace
    #   ([\w\s,]+)         The variables part (var1, var2, ...)
    #   ;\s*$              Semicolon until end of line
    var_decl_pattern = re.compile(r'^\s*([\w:\<>\*]+(?:\s+[\w:\<>\*]+)?)\s+([\w\s,]+);\s*$')
    
    for line in lines:
        line_stripped = line.strip()
        
        # Detect class start: "class Param {" or "class Param"
        # This also accounts for: "class Param : public Something {"
        class_match = re.match(r'^\s*class\s+(\w+)', line_stripped)
        if class_match:
            # If we already were in a class, we close it first
            if current_class is not None:
                # finalize old class if needed
                pass
            
            current_class = class_match.group(1)
            class_vars[current_class] = []
            in_class = True
            in_public_section = False
            continue
        
        # If we're inside a class, we look for a closing brace "};"
        if in_class:
            if '};' in line_stripped:
                # End of class
                in_class = False
                current_class = None
                in_public_section = False
                continue
            
            # Check if we hit 'public:' line
            if line_stripped.startswith('public:'):
                in_public_section = True
                continue
            
            # If we see 'private:' or 'protected:', we stop collecting
            if line_stripped.startswith('private:') or line_stripped.startswith('protected:'):
                in_public_section = False
                continue
            
            # If in the public section, try to match variable declarations
            if in_public_section:
                # Exclude lines that likely contain parentheses => function prototypes
                # e.g. "Param();" or "void foo(int x);" or anything that has '('
                if '(' in line_stripped or ')' in line_stripped:
                    continue
                
                # Attempt to match variable pattern
                match = var_decl_pattern.match(line_stripped)
                if match:
                    # group(1) = type, group(2) = var1, var2, ...
                    # For example: "int" and "a, b, c"
                    var_type = match.group(1)
                    var_names_str = match.group(2)
                    # Split by comma
                    var_names = [v.strip() for v in var_names_str.split(',')]
                    
                    # Store these variable names
                    for vn in var_names:
                        # Filter out any empty strings
                        if vn:
                            class_vars[current_class].append(vn)
    
    return class_vars


def generate_pybind_defines(class_vars):
    """
    Given a dict { className: [var1, var2, ...] },
    return a string that has lines of:
    .def_readwrite("varName", &ClassName::varName)
    for each className and variable.
    """
    lines = []
    for class_name, variables in class_vars.items():
        lines.append(f'py::class_<{class_name}>(m, "{class_name}")')
        # Optionally add constructor(s):
        lines.append(f'    .def(py::init<>())')
        # Add each variable
        for var in variables:
            lines.append(f'    .def_readwrite("{var}", &{class_name}::{var})')
        # close with semicolon
        lines[-1] += ";"  # append semicolon to last def
        # Alternatively, you could do lines.append("    ;") if you want a new line with semicolon
    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        print("Usage: parse_vars.py <filename.cpp/.h>")
        sys.exit(1)
    
    filename = sys.argv[1]
    class_vars = parse_cpp_for_class_vars(filename)
    result = generate_pybind_defines(class_vars)
    print(result)

if __name__ == "__main__":
    main()
