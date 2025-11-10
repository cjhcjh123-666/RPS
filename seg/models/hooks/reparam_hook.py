"""
结构重参数化Hook
在测试开始前自动进行重参数化融合
"""

from mmengine.hooks import Hook
from mmdet.registry import HOOKS


@HOOKS.register_module()
class ReparamHook(Hook):
    """结构重参数化Hook，在测试前自动融合多分支结构"""
    
    def __init__(self, 
                 auto_fuse_on_test=True,
                 auto_fuse_on_val=False,
                 verbose=True):
        """
        Args:
            auto_fuse_on_test (bool): 是否在测试开始前自动融合
            auto_fuse_on_val (bool): 是否在验证开始前自动融合  
            verbose (bool): 是否打印融合信息
        """
        self.auto_fuse_on_test = auto_fuse_on_test
        self.auto_fuse_on_val = auto_fuse_on_val
        self.verbose = verbose
        self._fused = False
    
    def before_test(self, runner):
        """测试前自动融合"""
        if self.auto_fuse_on_test and not self._fused:
            self._fuse_model(runner.model)
            
    def before_val(self, runner):
        """验证前自动融合"""
        if self.auto_fuse_on_val and not self._fused:
            self._fuse_model(runner.model)
    
    def _fuse_model(self, model):
        """执行模型融合"""
        # 处理可能被包装的模型 (DDP, DataParallel等)
        actual_model = model.module if hasattr(model, 'module') else model
        
        if self.verbose:
            print(f"ReparamHook: 模型类型: {type(model).__name__}")
            print(f"ReparamHook: 实际模型类型: {type(actual_model).__name__}")
            print(f"ReparamHook: 模型属性: {[attr for attr in dir(actual_model) if not attr.startswith('_')][:10]}...")
        
        if hasattr(actual_model, 'backbone'):
            backbone = actual_model.backbone
            backbone_type = type(backbone).__name__
            if self.verbose:
                print(f"ReparamHook: 检测到backbone类型: {backbone_type}")
                
            if hasattr(backbone, 'switch_to_deploy'):
                if self.verbose:
                    print("\n=== ReparamHook: 开始结构重参数化融合 ===")
                
                backbone.switch_to_deploy()
                self._fused = True
                
                if self.verbose:
                    print("=== ReparamHook: 重参数化融合完成 ===\n")
            else:
                if self.verbose:
                    print(f"ReparamHook: backbone {backbone_type} 不支持结构重参数化")
                    if hasattr(backbone, 'is_reparam'):
                        print(f"ReparamHook: backbone.is_reparam = {backbone.is_reparam}")
                    print(f"ReparamHook: backbone可用方法: {[m for m in dir(backbone) if 'switch' in m.lower() or 'deploy' in m.lower() or 'reparam' in m.lower()]}")
        else:
            if self.verbose:
                print("ReparamHook: 模型没有backbone属性")
                print(f"ReparamHook: 可用属性: {[attr for attr in dir(actual_model) if not attr.startswith('_')]}")
