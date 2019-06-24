"""Parses IP addresses and networks and extracts its properties."""
from h2oaicore.transformer_utils import CustomTransformer
import datatable as dt
import numpy as np
from abc import ABC, abstractmethod
import ipaddress


class IPAddressBaseTransformer(ABC):
    @staticmethod
    def get_default_properties():
        return dict(col_type="categorical", min_cols=1, max_cols=1, relative_importance=1)

    @abstractmethod
    def get_ip_property(self, value):
        raise NotImplementedError

    def parse_ipaddress(self, value):
        try:
            result = ipaddress.ip_address(value)
        except ValueError:
            result = ipaddress.ip_network(value)
        return result

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        return self.transform(X)

    def transform(self, X: dt.Frame):

        try:
            return X[:, {"x": (dt.isna(dt.f[0])) & None | self.get_ip_property(self.parse_ipaddress(dt.f[0]))}]
            # return X.to_pandas().astype(str).iloc[:, 0].apply(lambda x: self.get_ip_property(self.parse_ipaddress(x)))

        except ValueError:
            return np.zeros(X.shape[0])


class IPAddressAsIntegerTransformer(IPAddressBaseTransformer, CustomTransformer):
    def get_ip_property(self, value):
        ip_address_types = {'ipaddress.IPv4Address', 'ipaddress.IPv6Address'}
        ip_network_types = {'ipaddress.IPv4Network', 'ipaddress.IPv6Network'}
        if type(value) in ip_address_types:
            return int(value)
        elif type(value) in ip_network_types:
            return int(value[0])
        else:
            raise ValueError


class IsIPAddressMulticastTransformer(IPAddressBaseTransformer, CustomTransformer):
    def get_ip_property(self, value):
        return value.is_multicast


class IsIPAddressPrivateTransformer(IPAddressBaseTransformer, CustomTransformer):
    def get_ip_property(self, value):
        return value.is_private


class IsIPAddressGlobalTransformer(IPAddressBaseTransformer, CustomTransformer):
    def get_ip_property(self, value):
        return value.is_global


class IsIPAddressUnspecifiedTransformer(IPAddressBaseTransformer, CustomTransformer):
    def get_ip_property(self, value):
        return value.is_unspecified


class IsIPAddressReservedTransformer(IPAddressBaseTransformer, CustomTransformer):
    def get_ip_property(self, value):
        return value.is_reserved


class IsIPAddressLoopbackTransformer(IPAddressBaseTransformer, CustomTransformer):
    def get_ip_property(self, value):
        return value.is_loopback


class IsIPAddressLinkLocal(IPAddressBaseTransformer, CustomTransformer):
    def get_ip_property(self, value):
        return value.is_link_local
