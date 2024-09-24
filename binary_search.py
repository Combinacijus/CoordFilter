def binary_search(range_low, range_high, function, target, step_tolerance, print_debug=True):
    f_low = function(range_low)
    f_high = function(range_high)
    if f_low > f_high:
        swap_dir = True
    else:
        swap_dir = False
        
    if print_debug:
        str = f"| Target = {target:<.04f} |"
        str2 = "-"*len(str)
        print(f"{str2}\n{str}\n{str2}")
    
    while range_low <= range_high:
        mid = (range_low + range_high) / 2
        y = function(mid)

        if print_debug:
            print(f"low - high: {range_low:10.05f} - {range_high:<10.05f} | f({mid:9.05f})={function(mid):9.05f} | err: {(y-target):8.03f} | target: {target:<.03f} | step: {range_high - range_low:7.3f}|", end="")

        if abs(range_high - range_low) < step_tolerance or y == target:
            break
        elif y > target:
            if swap_dir is False:
                range_high = mid
            else:
                range_low = mid
            
            if print_debug:
                print(" HIGH")
        else:
            if swap_dir is False:
                range_low = mid
            else:
                range_high = mid
                
            if print_debug:
                print(" LOW")
                
    if print_debug:
        print(" FOUND")
        err = y - target
        str = f"| Error = {err:<.04f} ({err/y*100:.02f}%) | f({mid:.05f})={function(mid):.05f} |"
        str2 = "-"*len(str)
        print(f"{str2}\n{str}\n{str2}\n")
    return mid


if __name__ == "__main__":
    def func(x):
        return 2*x + 22

    target_percentage = 0
    target1 = target_percentage
    target2 = 100 - target_percentage
    min_offset = binary_search(-50, 50, func, target1, 0.01, True)
    max_offset = binary_search(-50, 50, func, target2, 0.01, True)
    
    print()
    print(f"f({min_offset}) = {func(min_offset)} -> ({target1} targeted)")
    print(f"f({max_offset}) = {func(max_offset)} -> ({target2} targeted)")