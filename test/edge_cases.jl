@testset "Edge cases" begin
    # 1. Minimum dimensions
    sig(zeros(2, 1), 1)  # D=1, M=1
    
    # 2. Single level
    sig(randn(10, 3), 1)  # M=1
    
    # 3. Test logsignature (you only tested signature)
    basis = prepare(3, 5)
    logsig(randn(20, 3), basis)
    
    # 4. Test 'sin' path type
    # (check_signatures.py has path_kind but only tested 'linear')
end