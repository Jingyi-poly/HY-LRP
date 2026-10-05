from abc import ABC, abstractmethod

class CutManager(ABC):
    """
    切割管理器抽象基类
    """

    def __init__(self, scen_tree):
        """
        初始化切割管理器

        Args:
            scen_tree: 场景树结构(字典格式 {stage: [nodes]})
        """
        self.scen_tree = scen_tree
        self.cuts = self._initialize_cut_storage()

    @abstractmethod
    def _initialize_cut_storage(self):
        """初始化切割存储数据结构"""
        pass

    @abstractmethod
    def add_cut(self, stage, node_idx, cut_data):
        """
        添加切割

        Args:
            stage: 阶段号
            node_idx: 节点索引
            cut_data: 切割数据（格式由子类定义）
        """
        pass

    @abstractmethod
    def get_cuts(self, stage, node_idx):
        """
        获取指定节点的所有切割

        Args:
            stage: 阶段号
            node_idx: 节点索引

        Returns:
            该节点的切割列表
        """
        pass

    def get_all_cuts(self):
        """返回所有切割"""
        return self.cuts
