"use client";

import { createContext, useContext, useEffect, useState } from "react";
import { useRouter, usePathname } from "next/navigation";

interface User {
    id: number;
    email: string;
    name: string;
    role: string;
    subjects: string[];
}

interface AuthContextType {
    user: User | null;
    token: string | null;
    login: (token: string, user: User) => void;
    logout: () => void;
    loading: boolean;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: React.ReactNode }) {
    const [user, setUser] = useState<User | null>(null);
    const [token, setToken] = useState<string | null>(null);
    const [loading, setLoading] = useState(true);
    const router = useRouter();
    const pathname = usePathname();

    useEffect(() => {
        // Load from local storage on mount
        const storedToken = localStorage.getItem("graide_token");
        const storedUser = localStorage.getItem("graide_user");

        if (storedToken && storedUser) {
            setToken(storedToken);
            setUser(JSON.parse(storedUser));
        } else {
            if (pathname !== "/login") {
                router.push("/login");
            }
        }
        setLoading(false);
    }, [pathname, router]);

    const login = (newToken: string, newUser: User) => {
        localStorage.setItem("graide_token", newToken);
        localStorage.setItem("graide_user", JSON.stringify(newUser));
        setToken(newToken);
        setUser(newUser);
        router.push("/");
    };

    const logout = () => {
        localStorage.removeItem("graide_token");
        localStorage.removeItem("graide_user");
        setToken(null);
        setUser(null);
        router.push("/login");
    };

    return (
        <AuthContext.Provider value={{ user, token, login, logout, loading }}>
            {!loading && children}
        </AuthContext.Provider>
    );
}

export function useAuth() {
    const context = useContext(AuthContext);
    if (context === undefined) {
        throw new Error("useAuth must be used within an AuthProvider");
    }
    return context;
}
